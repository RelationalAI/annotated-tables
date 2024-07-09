#  AYCT


"""
yes they are open. in fact I think we can clean this dataset by passing it through chatGPT and find which columns could be the target.
 Maybe run a simple logistic regression to verify that there is predictability.
 Lets say keep only the datasets that have AUC more than 0.7. We can give them the dataset to retrain TabPFN and get a publication :-)
"""
import string
from json import JSONDecodeError

import openai
from openai import OpenAI
import os
import time
import pandas
import torch
from sklearn.ensemble import GradientBoostingClassifier
from transformers import AutoConfig, AutoTokenizer, AutoModel
from sqlalchemy import text, create_engine, inspect
from sqlalchemy import create_engine, Column, Integer, String, Boolean, MetaData, Float
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
from sqlalchemy.exc import PendingRollbackError
from io import StringIO
import random

from wrapt_timeout_decorator import *

import subprocess
from tqdm import tqdm
import json
import torch.nn as nn
from torch.utils.data import Dataset, random_split
import re
from autogluon.tabular import TabularDataset

import sklearn

from toolkit import project_root

"""
1. get the schema of kaggle datasets
2. is there a target column that can be predicted?
3. simple tabular baseline: feedforward and bert 
"""
from pathlib import Path
import pickle


# def project_root():
#     return Path(__file__).parent.parent


class PickleCacheTool:
    def __init__(self):
        self.cache_dir = project_root() / "cache"

    def save(self, name, payload):
        with open(self.cache_dir / name, 'wb') as f:
            pickle.dump(payload, f)

    def load(self, name):
        with open(self.cache_dir / name, 'rb') as f:
            return pickle.load(f)

    def cache_exists(self, name):
        return (self.cache_dir / name).exists()


class Error403(Exception):
    pass


kaggle_path = project_root() / "data/kaggle"

Base = declarative_base()


class KaggleSqlRelSource:
    def __init__(self):
        self.catalog, self.sql_rel, self.manual_examples = self.load_exported_kaggle_sql_rel_source()
        sql_kws = ["group", "join", "where", 'like', 'in', 'avg', 'min', 'max', 'count',
                   'or', 'and', 'not', 'order', 'having',
                   'intersect', 'union', 'limit', 'except']
        keywords = {
            kw: kw for kw in sql_kws
        }
        keywords['group'] = "group by"
        keywords['order'] = "order by"
        self.sql_keywords = keywords

    @staticmethod
    def load_exported_kaggle_sql_rel_source():
        export_path = kaggle_path / "export2"
        catalog_path = export_path / "catalog.json"
        catalog = pandas.read_json(catalog_path)
        sql_rel_path = export_path / "sql_rel.json"
        sql_rel = pandas.read_json(sql_rel_path)
        manual_examples_path = export_path / "manual_prompt.json"
        manual_examples = pandas.read_json(manual_examples_path)
        return catalog, sql_rel, manual_examples

    def export_nirel_dataset_for_ast_parsing(self):
        # export all rel code and their execution flags
        # retain only syntactically correct code

        all_rel_code = []
        all_rel_code_executable = []
        for idx, sql_rel_row in self.sql_rel.iterrows():
            if sql_rel_row['rel_code'] is not None and sql_rel_row['rel_executable'] is not None:
                all_rel_code.extend(sql_rel_row['rel_code'])
                all_rel_code_executable.extend(sql_rel_row['rel_executable'])

        tenth = int(len(all_rel_code_executable) * 0.1)
        gen = torch.Generator()
        gen.manual_seed(42)
        train_len, valid_len, test_len = len(all_rel_code_executable) - tenth * 2, tenth, tenth
        indices = range(len(all_rel_code))
        train_indices, valid_indices, test_indices = torch.utils.data.random_split(
            indices, [train_len, valid_len, test_len],
            generator=gen
        )
        splits = {"train": train_indices,
                  "valid": valid_indices,
                  "test": test_indices}
        for k, indices in splits.items():
            code = [all_rel_code[i] for i in indices]
            executable = [all_rel_code_executable[i] for i in indices]
            splits[k] = (code, executable)

        return splits

    def export_nirel_dataset_for_ast_parsing_db_split(self):
        # export all rel code and their execution flags
        # retain only syntactically correct code
        gen = torch.Generator()
        gen.manual_seed(42)
        tenth = int(len(self.sql_rel) * 0.1)
        train_len, valid_len, test_len = len(self.sql_rel) - tenth * 2, tenth, tenth

        indices = range(len(self.sql_rel))
        train_indices, valid_indices, test_indices = torch.utils.data.random_split(
            indices, [train_len, valid_len, test_len],
            generator=gen
        )


        splits = {"train": train_indices,
                  "valid": valid_indices,
                  "test": test_indices}
        for k, indices in splits.items():
            code = []
            executable = []
            for i in indices:
                sql_rel_row = self.sql_rel.iloc[i]
                if sql_rel_row['rel_code'] is not None and sql_rel_row['rel_executable'] is not None:
                    code.extend(sql_rel_row['rel_code'])
                    executable.extend(sql_rel_row['rel_executable'])
            splits[k] = (code, executable)
        return splits

    def statistics_length(self):
        total_sql_code = 0
        for s in self.catalog.sql_translated:
            if s is not None:
                total_sql_code += len(s)

        total_valid_sql_code = 0
        for s in self.catalog.sql_executable:
            if s is not None:
                total_valid_sql_code += sum(s)

        total_rel_code = 0
        for s in self.sql_rel.rel_code:
            if s is not None:
                total_rel_code += len(s)

        total_correct_rel_code = 0
        for s in self.sql_rel.rel_executable:
            if s is not None:
                total_correct_rel_code += sum(s)

        total_rel_databases = 0
        for s in self.sql_rel.rel_executable:
            if s is not None and any(s):
                total_rel_databases += 1

        sql_executable_dbs = 0
        for row in self.catalog.iloc:
            sql_executable = row['sql_executable']
            if sql_executable is not None and any(sql_executable):
                sql_executable_dbs += 1

    def categorize_synthesized(self):
        all_correct = []
        syntax = []
        semantics = []
        for row in self.sql_rel.iloc:
            rels, rels_correct = row['rel_code'], row['rel_executable']
            sqls = row['sql_code']
            infos = row['rel_executable_info']
            if rels is not None and rels_correct is not None:
                for rel, correct, sql, info in zip(rels, rels_correct, sqls, infos):
                    if correct:
                        all_correct.append((rel, sql))
                    if info == 'syntax':
                        syntax.append((rel, sql))
                    if info == 'comparison':
                        semantics.append((rel, sql))
        return all_correct, syntax, semantics

    def examples_synthesized(self):
        all_correct, syntax, semantics = self.categorize_synthesized()
        correct_sample = random.sample(all_correct, 100)
        syntax_sample = random.sample(syntax, 100)
        semantics_sample = random.sample(semantics, 100)
        return correct_sample, syntax_sample, semantics_sample

    def classified_accuracy(self):
        all_count = 0
        all_correct = 0
        cls_count = {k: 0 for k in self.sql_keywords.values()}
        cls_correct = {k: 0 for k in self.sql_keywords.values()}
        for row in tqdm(self.sql_rel.iloc, total=len(self.sql_rel)):
            rels, rels_correct = row['rel_code'], row['rel_executable']
            sqls = row['sql_code']
            infos = row['rel_executable_info']
            if rels is not None and rels_correct is not None:
                for rel, correct, sql, info in zip(rels, rels_correct, sqls, infos):
                    clss = self.classify_sql(sql[1])
                    for c in clss:
                        cls_count[c] += 1
                        if correct:
                            cls_correct[c] += 1
                    if correct:
                        all_correct += 1
                    all_count += 1
        cls_percent = {}
        for k in cls_count:
            cls_percent[k] = cls_correct[k] / cls_count[k]

        dat = {
            'type': [],
            'sql count': [],
            "rel correct": [],
            'rel correct percentage': []
        }
        for k in self.sql_keywords.values():
            dat['type'].append(k)
            dat['sql count'].append(cls_count[k])
            dat['rel correct'].append(cls_correct[k])
            dat['rel correct percentage'].append(cls_percent[k])

        df = pandas.DataFrame(dat)
        df = df.sort_values(by="sql count", ascending=False)
        df.reset_index(drop=True, inplace=True)
        df['rel correct percentage'] = df['rel correct percentage'] * 100

        tex = df.to_latex(index=False)
        print(tex)

    def classify_sql(self, sql):
        cls = []
        sql = sql.lower().split()

        for kw, cl in self.sql_keywords.items():
            if kw in sql:
                cls.append(cl)
        return cls


class ChatGPTSQLResponse(Base):
    __tablename__ = 'chatgpt_sql'
    kaggle_ref = Column(String, primary_key=True)
    openai_response = Column(String)


class ChatGPTSQLToRELResponse(Base):
    __tablename__ = 'chatgpt_sql_to_rel'
    kaggle_ref = Column(String, primary_key=True)
    openai_response = Column(String)


class KaggleAPIResponse(Base):
    __tablename__ = 'kaggle_response'
    min_size = Column(Integer, primary_key=True)
    max_size = Column(Integer)
    page = Column(Integer)
    kaggle_response = Column(String)


class ExecutableFlag(Base):
    __tablename__ = 'executable_flag'
    ref = Column(String, primary_key=True)
    executed = Column(Boolean)
    executable_flag = Column(String)


class RelExecution(Base):
    __tablename__ = 'rel_execution'
    ref = Column(String, primary_key=True)
    acc = Column(String)
    infos = Column(String)


class RelExecutionRerun(Base):
    __tablename__ = 'rel_execution_rerun'
    ref = Column(String, primary_key=True)
    acc = Column(String)
    infos = Column(String)
    sql_time = Column(String)
    rel_time = Column(String)
    problems = Column(String)


class SQLMagic:
    max_rows = 1000

    def __init__(self, name=None):
        if name is None:
            name = 'sqlite://'
        else:
            name = f'sqlite://{name}'
        self.engine = create_engine(name, echo=False)
        self.insp = None

    def terminate(self):
        self.engine.dispose()

    def remove_all_tables_terminate(self):
        metadata = MetaData(bind=self.engine)
        metadata.reflect()
        metadata.drop_all()
        self.engine.dispose()

    def rename_columns(self, columns):
        mapping = {}
        for column in columns:
            split_by = "-", "_", None
            new_column = column
            for by in split_by:
                new = new_column.split(by)
                for uidx, u in enumerate(new):
                    if len(u) != 0:
                        up = u[0].upper() + u[1:]
                    else:
                        up = ""
                    new[uidx] = up
                new_column = "".join(new)

            # new_column = new_column.replace("%", "")
            # new_column = new_column.replace("/", "")
            # new_column = new_column.replace("(", "")
            new_column = new_column.replace(")", "")
            chars = re.escape(string.punctuation)
            new_column = re.sub('[' + chars + ']', '', new_column)

            mapping[column] = new_column
        return mapping

    def csv_to_sqlite(self, table_name="diabetes", csv_path=project_root() / 'diabetes.csv'):
        df = pandas.read_csv(csv_path)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        df = df.rename(columns=self.rename_columns(df.columns))
        df = df.iloc[:SQLMagic.max_rows]

        df.to_sql(table_name, con=self.engine, index=False, if_exists='replace', index_label=None)

    def get_all_tables_schema(self, string=True):
        tables = self.get_table_names()
        all_schema = []
        raw_schemas = {}
        for table in tables:
            with self.engine.connect() as conn:
                cursor = conn.execute(text(f"pragma table_info({table});"))
                headers = cursor._metadata.keys
                rows = cursor.fetchall()
                schema, raw_schema = self.make_schema(table, rows, headers)
                raw_schemas[table] = raw_schema
                all_schema.append(schema)
        if string:
            all_schema = "\n".join(schema)
        return all_schema, raw_schemas

    def get_all_tables_first_row_data(self):
        tables = self.get_table_names()
        first_row_data = []
        for table in tables:
            first_row = {}
            with self.engine.connect() as conn:
                cursor = conn.execute(text(f"SELECT * FROM {table} ORDER BY ROWID ASC LIMIT 1"))
                headers = cursor._metadata.keys
                rows = cursor.fetchall()
                if len(rows) > 0:
                    for header, col in zip(headers, rows[0]):
                        first_row[header] = col
            first_row_data.append(json.dumps(first_row))
        return first_row_data

    def make_schema(self, table_name, rows, headers):
        schema = f"CREATE TABLE {table_name}("
        for row in rows:
            schema += f"\n    {row[1]}  {row[2]},"

        schema = schema[:-1]
        schema += "\n);"
        return schema, [(row[1], row[2]) for row in rows]

    def get_table_names(self):
        if self.insp is None:
            self.insp = inspect(self.engine)
        return self.insp.get_table_names()

    def execute_query(self, query):
        with self.engine.connect() as conn:
            cursor = conn.execute(text(query))
            headers = cursor._metadata.keys
            ret = cursor.fetchall()
            return ret, headers

    def execute_queries(self, queries, **kwargs):
        results = []
        with self.engine.connect() as conn:
            for query in queries:
                try:
                    cursor = conn.execute(text(query))
                    headers = cursor._metadata.keys
                    ret = cursor.fetchall()
                    r = ret, list(headers)
                    results.append(r)
                except (Exception,) as e:
                    results.append(None)
        return results

    @timeout(120)
    def execute_query_with_timeout(self, queries, **kwargs):
        return self.execute_queries(queries, **kwargs)

    def load_sqlite_dataset(self, sqlite_files):
        # too few tables have sqlite files
        # 0.117 %
        pass

    def load_csv_dataset(self, csv_files):
        for csv_file in csv_files:
            tbl_name = csv_file.stem
            tbl_name = self.table_name_sanitize(tbl_name)
            self.csv_to_sqlite(table_name=tbl_name, csv_path=csv_file)

    def table_name_sanitize(self, orig_name):
        split_chars = [" ", "_", "-"]
        name = orig_name
        for c in split_chars:
            splitted = name.split(c)
            upp = []
            for s in splitted:
                if len(s) != 0:
                    upp.append(s[0].upper() + s[1:])
            name = "".join(upp)
        return name

    def export_table(self):
        # if not table_id.startswith('table'):
        #     table_id = 'table_{}'.format(table_id.replace('-', '_'))
        table_info = \
            self.conn.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[0].as_dict()[
                "sql"]
        schema_str = schema_re.findall(table_info)[0]
        schema = {}
        for tup in schema_str.split(', '):
            c, t = tup.split()
            schema[c] = t

        query = 'SELECT * FROM {}'.format(table_id)
        out = self.conn.query(query)
        return schema, out.as_dict()


class KaggleDownloader:
    """
    Download Kaggle datasets and catalog them

    Before running the code, set up Kaggle API first
    https://www.kaggle.com/docs/api
    """
    configs = {
        "1": {
            "max_file_size": 10000000,
            "min_file_size": 100000,
            "min_usability": 0.1
        },
        "2": {
            "max_file_size": 1000000000,
            "min_file_size": 10000000,
            "min_usability": 0.1
        }
    }

    def __init__(self, subset_index="1"):
        self.kaggle_command = os.getenv('KAGGLE_PATH')
        if self.kaggle_command is None:
            self.kaggle_command = "kaggle"
        self.datasets = []
        self.pickleTool = PickleCacheTool()
        self.subset_index = subset_index
        self.dataset_catalog_path = kaggle_path / "set" / subset_index / "datasets_catalog.csv"
        self.dataset_catalog_path.parent.mkdir(parents=True, exist_ok=True)

        # controls the dataset size
        config = KaggleDownloader.configs[subset_index]
        self.max_file_size = config["max_file_size"]
        self.min_file_size = config["min_file_size"]
        self.min_usability = config['min_usability']
        self.kaggle_cache = self.connect_sqlite_api_call_cache_db()

        self.total_max_row = 70000

        self.df = None

    def connect_sqlite_api_call_cache_db(self):
        engine = create_engine(f"sqlite:///{kaggle_path}/kaggle_cache.db")
        # Create the table in the database
        Base.metadata.create_all(engine)

        # Create a session to interact with the database
        Session = sessionmaker(bind=engine)
        Session = scoped_session(Session)
        session = Session()
        try:
            _ = session.connection()
        except PendingRollbackError:
            session.rollback()
        return session

    def load_api_call_cache_db(self, min_size, max_size, page):
        res = self.kaggle_cache.query(KaggleAPIResponse).filter_by(min_size=min_size, max_size=max_size,
                                                                   page=page).first()
        if res is None:
            return None
        else:
            res = json.loads(res.openai_response)
            return res

    def write_api_call_cache_db(self, min_size, max_size, page, res):
        exist = self.kaggle_cache.query(KaggleAPIResponse).filter_by(min_size=min_size, max_size=max_size,
                                                                     page=page).first()
        if exist is None:
            res_str = json.dumps(res)
            new_response = KaggleAPIResponse(min_size=min_size, max_size=max_size, page=page, kaggle_response=res_str)
            self.kaggle_cache.add(new_response)
            self.kaggle_cache.commit()

    def rm_row_api_all_cache_db(self, min_size, max_size, page):
        res = self.kaggle_cache.query(KaggleAPIResponse).filter_by(min_size=min_size, max_size=max_size,
                                                                   page=page).first()
        self.kaggle_cache.delete(res)
        self.kaggle_cache.commit()

    def subset_path(self):
        return kaggle_path / "set" / self.subset_index

    def run_command(self, str_command, retry=10):
        while retry > 0:
            command = str_command.split()
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)
            out = result.stdout
            error = result.stderr
            if "403 - Forbidden" in out:
                raise Error403
            if "404 - Not Found" in out:
                raise Error403
            if "error: unrecognized arguments:" in error:
                raise Error403
            if "EOFError" in error or "EOFError" in out:
                raise Error403
            if error != "" or result.returncode != 0:
                time.sleep(10)
                retry -= 1
            else:
                return result, out, error
        raise RuntimeError(out + error)

    def query_dataset_page(self, df, next_max_size, next_min_size, page=1):
        cache = self.load_api_call_cache_db(next_min_size, next_max_size, page)
        if cache is None:
            command = f"{self.kaggle_command} datasets list --max-size {next_max_size} --min-size {next_min_size} --csv -p {page}"
            result, out, error = self.run_command(command)
            if "No datasets found" in out:
                raise StopIteration
            else:
                self.write_api_call_cache_db(next_min_size, next_max_size, page, out)
        else:
            out = cache
        new_df = pandas.read_csv(StringIO(out))
        df = pandas.concat([df, new_df])
        return df

    def progressive_query_range_determination(self, max_size, prev_max_size, min_offset):
        """
        Kaggle API query does not return more than 10000 results.
        We query the size intervals repeatedly. Each size interval will have no more than 10000 results.

        :param max_size: The overall max size
        :param prev_max_size:
        :return:
        """
        mmax = max_size
        while True:
            if max_size - prev_max_size < min_offset:
                next_max_size = mmax
            else:
                next_max_size = (mmax + prev_max_size) // 2
            command = f"{self.kaggle_command} datasets list --max-size {next_max_size} --min-size {prev_max_size} --csv -p 500"
            result, out, error = self.run_command(command)
            if "No datasets found" in out:
                return next_max_size, prev_max_size
            else:
                mmax = next_max_size

    def all_catalog_datasets(self):
        dfs = []
        for subset_index in KaggleDownloader.configs.keys():
            df_path = kaggle_path / "set" / subset_index / "datasets_catalog.csv"
            df = pandas.read_csv(df_path)
            if 'Unnamed: 0' in df.columns:
                df = df.drop('Unnamed: 0', axis=1)
            dfs.append(df)

        df = pandas.concat(dfs)
        df = df.iloc[:self.total_max_row]
        return df

    def byte_size_sequence(self, size_sequence):
        for s in size_sequence:
            nm = int(re.findall("[0-9]+", s)[0])
            unit = re.findall("[A-Z]+", s)[0]

    def catalog_datasets(self):
        if self.df is not None:
            return self.df

        if self.dataset_catalog_path.exists():
            df = pandas.read_csv(self.dataset_catalog_path)
        else:
            max_size = self.max_file_size
            min_size = self.min_file_size
            min_offset = (max_size - min_size) // 100
            next_min_size = min_size
            df = pandas.DataFrame()
            while True:
                next_max_size, next_min_size = \
                    self.progressive_query_range_determination(max_size, next_min_size, min_offset)
                new_df = self.get_all_datasets_in_size_range(next_max_size, next_min_size)
                df = pandas.concat([df, new_df])
                if next_max_size == max_size:
                    break
                next_min_size = next_max_size
            df.to_csv(self.dataset_catalog_path)
        df = df[df['usabilityRating'] > self.min_usability]
        try:
            df = df.drop("Unnamed: 0", axis=1)
        except:
            pass
        self.df = df
        return df

    def get_all_datasets_in_size_range(self, next_max_size, next_min_size):
        print(f'Cataloging datasets in range {next_min_size} to {next_max_size}')
        rows = 0
        next_page = 1

        df = pandas.DataFrame()
        pbar = tqdm()
        while True:
            try:
                df = self.query_dataset_page(df, next_max_size, next_min_size, next_page)
                next_page += 1
                new_rows = len(df)
                pbar.update(new_rows - rows)
                rows = new_rows
            except RuntimeError:
                time.sleep(10)
            except StopIteration:
                break
        return df

    def get_dataset_directory(self, name):
        datasets_path = kaggle_path / 'datasets'
        dir_name = self.sanitize_linux_directory_name(name)
        path = datasets_path / dir_name
        return path

    def download_datasets(self, step):
        if step > 1:
            pass
        else:
            catalog = self.all_catalog_datasets()
            datasets_path = kaggle_path / 'datasets'
            errored_file = kaggle_path / "errors.json"
            with open(errored_file, 'r') as f:
                error_ds = json.load(f)
            for row in tqdm(catalog.iloc, total=len(catalog), desc="Downloading Kaggle datasets"):
                name = row['ref']
                dir_name = self.sanitize_linux_directory_name(name)
                path = datasets_path / dir_name
                if not (path.exists() or name in error_ds):
                    path = path.absolute()
                    command = f"{self.kaggle_command} datasets download -d {name} --path {path} --unzip -q"
                    try:
                        self.run_command(command)
                    except Error403:
                        error_ds.append(name)
                        with open(errored_file, 'w') as f:
                            error_dump = json.dumps(error_ds)
                            f.write(error_dump)

    def sanitize_linux_directory_name(self, name):
        return name.replace("/", "_")

    def find_files(self, name):
        """
        Kaggle dataset directory is unstructured. The data file can be of any type,
        in any directory.

        :return:
        """
        path = self.get_dataset_directory(name)
        csvs = list(path.glob("**/*.csv"))
        sqlite = list(path.glob("**/*.sqlite"))
        return csvs, sqlite


class AdapterKaggleSQLRel:
    def __init__(self):
        self.exported_kaggle = KaggleSqlRelSource()
        self.kaggle_downloader = KaggleDownloader()

    def get_one_table_schema(self, i):
        row = self.exported_kaggle.catalog.iloc[i]
        schema = row['schemas']
        # use only one table's schema
        if schema is None:
            return None
        schema = schema[0]
        return schema

    def ref_to_row(self, ref):
        row = self.exported_kaggle.catalog[self.exported_kaggle.catalog['ref'] == ref]
        return row

    def ref_to_iloc(self, ref):
        row = self.ref_to_row(ref)
        return row.index.item()

    def get_catalog_row(self, i):
        row = self.exported_kaggle.catalog.iloc[i]
        return row

    def get_table_data(self, name):
        csvs, sqlites = self.kaggle_downloader.find_files(name)
        dfs = {}
        for csv_file in csvs:
            tbl_name = csv_file.stem
            df = pandas.read_csv(csv_file)
            dfs[tbl_name] = df
            # use only one table
            break
        return dfs


class ChatGPTSuggestTargetColumn:
    def __init__(self):
        # Load your API key from an environment variable or secret management service
        # api_key = os.getenv("OPENAI_API_KEY")

        # Initialize the OpenAI API client
        self.client = OpenAI()

    def prompt_v1(self, schema):
        prompt = (f"Consider a machine learning model that takes a few numeric input columns and predict "
                  "a single classification target column. Given the following schema of a data table, suggest "
                  "the input columns and target column, such that the target may be predicted"
                  "from the inputs non-trivially. \n"
                  f"Schema: {schema}\n"
                  f"Respond in JSON format with 'input_columns' and 'output_column'.")
        return prompt

    def query_chat_gpt(self, query, num_results=1, return_json=True, **kwargs):
        retry = 3

        while True:
            try:
                if return_json:
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",  # "gpt-3.5-turbo",  # $0.002 / 1k
                        # model="gpt-4",  # $0.03 / 1k
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": f"{query}"},
                        ],
                        response_format={"type": "json_object"},
                        n=num_results,
                        **kwargs
                    )
                else:
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",  # "gpt-3.5-turbo",  # $0.002 / 1k
                        # model="gpt-4",  # $0.03 / 1k
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": f"{query}"},
                        ],
                        n=num_results,
                        **kwargs
                    )
                if num_results == 1:
                    return response.choices[0].message.content
                else:
                    rets = []
                    for i in range(num_results):
                        ret = response.choices[i].message.content
                        rets.append(ret)
                    return rets
            except openai.APIError:
                retry -= 1
                if retry == 0:
                    raise

    async def query_chat_gpt_async(self, query, model="gpt-3.5-turbo-16k", **kwargs):
        response = await openai.ChatCompletion.acreate(
            model=model,  # "gpt-3.5-turbo",  # $0.002 / 1k
            # model="gpt-4",  # $0.03 / 1k
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{query}"},

            ],
            response_format={"type": "json_object"},
            n=1,
            **kwargs
        )

        ret = response['choices'][0]['message']['content']
        return ret


class TabularBaselineModel(nn.Module):
    def __init__(self, args, verbose=True):
        super(TabularBaselineModel, self).__init__()
        self.args = args
        self.tokenizer = None
        self.config = AutoConfig.from_pretrained("microsoft/CodeGPT-small-py")
        self.config.use_cache = False
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py")
        self.model = AutoModel.from_pretrained("microsoft/CodeGPT-small-py", use_cache=False)

    def forward(self, inputs, labels):
        pass

    def num_parameters(self):
        return self.model.num_parameters()

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()


from autogluon.tabular import TabularPredictor


class AutoG:
    def __init__(self):
        self.sql = SQLMagic()

    def predict(self, data, suggested_columns):
        assert len(data) == 1, "We do not support join tables in prediction"
        for k, tbl in data.items():
            break

        renamed = self.sql.rename_columns(tbl.columns)
        tbl = tbl.rename(columns=renamed)
        tbl = tbl.dropna()

        input_columns = suggested_columns['input_columns']
        selected_columns = input_columns + [suggested_columns['output_column']]
        try:
            selected_input = tbl[selected_columns]
        except KeyError:
            raise ValueError

        training_cnt = int(len(selected_input) * 0.8)
        training_input, validation_input = (
            random_split(range(len(selected_input)), (training_cnt, len(selected_input) - training_cnt),
                         generator=torch.Generator().manual_seed(42)))
        selected_input = selected_input.reset_index(drop=True)
        training_input = selected_input.iloc[training_input.indices].copy()
        validation_input = selected_input.iloc[validation_input.indices].copy()

        predictor = TabularPredictor(label=suggested_columns['output_column'],
                                     verbosity=1).fit(training_input,
                                                      validation_input,
                                                      presets="medium_quality",
                                                      time_limit=30)
        summary = predictor.fit_summary()
        problem_type = predictor.problem_type
        if predictor.problem_type == "regression":
            emetric = 'r2'
            leaderboard = predictor.leaderboard(validation_input, extra_metrics=['r2'])
        elif predictor.problem_type == "binary":
            emetric = 'roc_auc'
            leaderboard = predictor.leaderboard(validation_input, extra_metrics=['roc_auc'])
        elif predictor.problem_type == "multiclass":
            emetric = 'roc_auc'
            leaderboard = predictor.leaderboard(validation_input, extra_metrics=['roc_auc'])
        # elif predictor.problem_type == "quantile":
        #     pass
        else:
            raise

        uniform_score = leaderboard.iloc[0][emetric]
        model = leaderboard.iloc[0].model

        return summary, leaderboard, problem_type, uniform_score, emetric, model


class TabularFactory:
    @staticmethod
    def _create_dataset(data, suggested_columns, to_string=True):
        for k, table in data.items():
            break
        u = int(len(table) * 0.1)
        valid, test, train = random_split(range(len(table)), [u, u, len(table) - 2 * u],
                                          generator=torch.Generator().manual_seed(42))
        ds = {
            "train": train,
            "valid": valid,
            "test": test
        }
        input_columns = suggested_columns["input_columns"]
        output_column = suggested_columns["output_column"]
        for k, indices in ds.items():
            slice = table.iloc[indices.indices]
            inputs = slice[input_columns]
            output = slice[output_column]
            ds[k] = (inputs, output)
        if to_string:
            for split, slices in ds.items():
                str_inputs = []
                for i in slices[0].iloc:
                    ls = i.values.tolist()
                    str_input = [str(l) for l in ls]
                    str_input = "; ".join(str_input)
                    str_inputs.append(str_input)
                str_outputs = []
                for o in slices[1].iloc:
                    str_outputs.append(str(o))
                ds[split] = list(zip(str_inputs, str_outputs))

        return ds

    @staticmethod
    def create_dataset_for_trainer(data, suggested_columns):
        return TabularFactory._create_dataset(data, suggested_columns, to_string=True)

    @staticmethod
    def create_dataset_for_boost(data, suggested_columns):
        return TabularFactory._create_dataset(data, suggested_columns, to_string=False)

    @staticmethod
    def create_dataset_for_autogluon(data, suggested_columns):
        ds = TabularFactory._create_dataset(data, suggested_columns, to_string=False)
        for k, v in ds.items():
            pass
            # make data frames and pass on columns
        return


class Boost:
    def __init__(self, dataset):
        x_train, y_train = list(zip(*dataset['train']))
        x_valid, y_valid = list(zip(*dataset['valid']))
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                         max_depth=1, random_state=0).fit(x_train, y_train)
        clf.score(x_valid, y_valid)


class CollatorForTabularPrototype:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input, output = zip(*features)
        input = self.tokenizer.batch_encode_plus(input, max_length=512, padding=True, truncation=True,
                                                 return_tensors='pt', )
        output = self.tokenizer.batch_encode_plus(output, max_length=512, padding=True, truncation=True,
                                                  return_tensors='pt', )
        return {
            "inputs": input,
            "labels": output
        }


Base = declarative_base()


class InputTargetCache(Base):
    __tablename__ = 'CacheOne'
    key = Column(String, primary_key=True)
    payload = Column(String)


class EvaluatedCache2(Base):
    __tablename__ = 'EvaluatedCache2'
    key = Column(String, primary_key=True)
    payload = Column(String)


class SQLCacher:

    def __init__(self):
        self.engine_address = f"sqlite:///{kaggle_path}/tabpfn_cache.db"
        self.kaggle_cache = self.connect_sqlite_api_call_cache_db()
        # self.total_max_row = 70000
        self.df = None

    def connect_sqlite_api_call_cache_db(self):
        engine = create_engine(self.engine_address)
        # Create the table in the database
        Base.metadata.create_all(engine)

        # Create a session to interact with the database
        Session = sessionmaker(bind=engine)
        Session = scoped_session(Session)
        session = Session()
        try:
            _ = session.connection()
        except PendingRollbackError:
            session.rollback()
        return session

    def load_cache(self, DB, key):
        res = self.kaggle_cache.query(DB).filter_by(key=key).first()
        if res is None:
            return None
        else:
            res = json.loads(res.payload)
            return res

    def write_cache(self, DB, key, payload, overwrite=False):
        exist = self.kaggle_cache.query(DB).filter_by(key=key).first()
        if exist and overwrite:
            self.kaggle_cache.delete(exist)
        if exist is None or overwrite:
            res_str = json.dumps(payload)
            new_response = DB(key=key, payload=res_str)
            self.kaggle_cache.add(new_response)
            self.kaggle_cache.commit()


class FeasibilityPrototype:
    def __init__(self, load=True, start=None, end=None):
        self.disk_path = project_root() / "data/classification"
        self.disk_path.mkdir(exist_ok=True)
        self.version = "prototype7"
        if start is None and end is None:
            self.selected_indices = list(range(0, 70000))
        else:
            self.selected_indices = list(range(start, end))

        self.cacher = SQLCacher()

        self.adapter = AdapterKaggleSQLRel()
        self.suggest = ChatGPTSuggestTargetColumn()
        self.suggested_columns = None
        self.suggest_input_target(load=load)

    def suggest_input_target(self, load=True):
        dump_path = self.disk_path / f"suggested_input_target_{self.version}.json"
        if load and dump_path.exists():
            with open(dump_path, 'r') as f:
                columns = json.loads(f.read())
        else:
            selected_data_bases = self.selected_data_bases()
            raw_answers = []
            columns = {}
            for name, i in tqdm(selected_data_bases):
                res = self.cacher.load_cache(InputTargetCache, name)
                if res is None:
                    try:
                        schema = self.adapter.get_one_table_schema(i)
                    except IndexError:
                        continue
                    if schema is None:
                        continue
                    prompt = self.suggest.prompt_v1(schema)
                    try:
                        answer = self.suggest.query_chat_gpt(prompt)
                        json.loads(answer)
                    except:
                        continue
                    self.cacher.write_cache(InputTargetCache, name, answer)
                else:
                    answer = res
                raw_answers.append(answer)
                try:
                    columns[name] = json.loads(answer)
                except JSONDecodeError:
                    pass
            with open(dump_path, 'w') as f:
                f.write(json.dumps(columns))
        self.suggested_columns = columns
        return columns

    def selected_data_bases(self):
        selected_data_bases_range = self.selected_indices
        names = []
        for i in selected_data_bases_range:
            row = self.adapter.get_catalog_row(i)
            name = row['ref']
            names.append(name)
        return tuple(zip(names, selected_data_bases_range))

    def kaggle_iterator(self):
        for name, i in self.selected_data_bases():
            try:
                schema = self.adapter.get_one_table_schema(i)
            except IndexError:
                yield None
                continue
            if schema is None:
                yield None
                continue
            try:
                suggested_columns = self.suggested_columns[name]
                data = self.adapter.get_table_data(name)
                row = self.adapter.get_catalog_row(i)
                yield schema, suggested_columns, data, row
                continue
            except:
                yield None
                continue


def dev_demo_query_chat_gpt():
    adapter = AdapterKaggleSQLRel()
    schema = adapter.get_one_table_schema(5)
    suggest = ChatGPTSuggestTargetColumn()
    prompt = suggest.prompt_v1(schema)
    answer = suggest.query_chat_gpt(prompt)


def run_main():
    feasi = FeasibilityPrototype()
    # feasi.suggest_input_target(load=False)
    feasi.run_main()


def dev_make_dataset():
    feasi = FeasibilityPrototype()
    it = feasi.kaggle_iterator()
    for schema, suggested_columns, data, row in it:
        d = TabularFactory.create_dataset(data, suggested_columns)


def boost_sweep():
    feasi = FeasibilityPrototype()
    it = feasi.kaggle_iterator()
    for schema, suggested_columns, data, row in it:
        dataset = TabularFactory.create_dataset(data, suggested_columns, to_string=False)
        boost = Boost(dataset)


def autogluon_sweep():
    start_time = time.time()
    feasi = FeasibilityPrototype()
    it = feasi.kaggle_iterator()
    autog = AutoG()
    all_results = {
        "dataset": [],
        "input columns": [],
        "target column": [],
        "problem type": [],
        "metric": [],
        "score": []
    }
    for schema, suggested_columns, data, row in it:
        try:
            res = autog.predict(data, suggested_columns)
            summary, leaderboard, problem_type, uniform_score, emetric, model = res
            all_results['dataset'].append(row['ref'])
            all_results['input columns'].append(" ".join(suggested_columns['input_columns']))
            all_results['target column'].append(suggested_columns["output_column"])
            all_results['problem type'].append(problem_type)
            all_results['model'].append(model)
            all_results['metric'].append(emetric)
            all_results['score'].append(uniform_score)
        except:
            pass
    df = pandas.DataFrame(all_results)
    print(df)
    print(df.to_latex())
    df.to_csv(project_root() / "proto_tabpfn_baselines.csv")
    print(f"Time elapsed: {time.time() - start_time}")


# if __name__ == '__main__':
#     autogluon_sweep()
if __name__ == '__main__':
    FeasibilityPrototype()
