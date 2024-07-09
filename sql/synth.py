import asyncio
import concurrent
import csv
import io
import json
import os
import random
import sqlite3
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

from queue import Queue
from urllib.error import HTTPError

import numpy as np
import pandas
import sqlalchemy
from sqlalchemy.exc import PendingRollbackError
from tqdm import tqdm
from wrapt_timeout_decorator import *

from sqlalchemy import Column, Integer, String, Boolean, MetaData
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session

from llm.prompting import WikiSQLPrompting
from rel.executor import RELX
from toolkit.util import project_root, PickleCacheTool
import torch
import gzip
from sqlalchemy import text, create_engine, inspect
from llm.chatgpt import ChatGPT
import re
import string

kaggle_path = project_root() / "data/kaggle"

import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Error403(Exception):
    pass


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
        session = scoped_session(Session)
        # session = Session()
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


class KaggleSynthDataset(torch.utils.data.Dataset):
    def __init__(self, split="train"):
        super().__init__()
        self.ds = None
        path = self.get_path(split)
        with gzip.open(path, 'rt') as f:
            st = f.read()
        self.ds = json.loads(st)

    @staticmethod
    def get_path(split):
        path = project_root() / f"data/kaggle/kaggle_chatgpt_synthesis_{split}.json"
        return path

    def __getitem__(self, item):
        return self.ds[item]

    def __len__(self):
        return len(self.ds)


# folding these main methods
if True:
    def main():
        conn = sqlite3.connect('database.db')
        cur = conn.cursor()
        with open(project_root() / 'diabetes.csv') as f:
            reader = csv.reader(f)
            data = list(reader)


    def main2():
        engine = create_engine('sqlite://', echo=False)
        df = pandas.read_csv(project_root() / 'diabetes.csv')
        df.to_sql('diabetes', con=engine, index=False, if_exists='replace', index_label=None)
        with engine.connect() as conn:
            cursor = conn.execute(text("pragma table_info(diabetes);"))
            # ret = conn.execute(text("SELECT * FROM diabetes")).fetchall()
            # ret = conn.execute(text("SELECT * FROM diabetes WHERE Pregnancies>10")).fetchall()
            headers = cursor._metadata.keys
            ret = cursor.fetchall()
            print(ret)


    def persistent_test():
        engine = create_engine('sqlite://', echo=False)
        # df = pandas.read_csv(project_root() / 'diabetes.csv')
        # df.to_sql('diabetes', con=engine, index=False, if_exists='replace', index_label=None)
        with engine.connect() as conn:
            cursor = conn.execute(text("pragma table_info(diabetes);"))
            headers = cursor._metadata.keys
            ret = cursor.fetchall()
            print(ret)


    class NoDataFound(Exception):
        pass


def prototytpe_chat_gpt_sql_execution_performance():
    sql = SQLMagic()
    sql.csv_to_sqlite()
    all_schema = sql.get_all_tables_schema()
    print(all_schema)
    queries = ["""SELECT Outcome, AVG(BMI) AS AvgBMI, AVG(Glucose) AS AvgGlucose
FROM diabetes
GROUP BY Outcome
ORDER BY AvgBMI DESC;
    """,
               """SELECT Pregnancies, AVG(Age) AS AvgAge
        FROM diabetes
        WHERE BMI > (SELECT AVG(BMI) FROM diabetes)
        GROUP BY Pregnancies;
        """,
               """SELECT (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM diabetes)) AS Percentage
        FROM diabetes
        WHERE Glucose > (SELECT AVG(Glucose) FROM diabetes);
               """
               ]
    sql.execute_query(queries[2])

    """
    The synthesis quality is bonkers. Great quality. Great results.
    https://chat.openai.com/share/813a1f19-9b26-4127-8e2b-ec2156bf6d03
    """


Base = declarative_base()


class ChatGPTSQLResponse(Base):
    __tablename__ = 'chatgpt_sql'
    kaggle_ref = Column(String, primary_key=True)
    openai_response = Column(String)


class ChatGPTSQLToRELResponse(Base):
    __tablename__ = 'chatgpt_sql_to_rel'
    kaggle_ref = Column(String, primary_key=True)
    openai_response = Column(String)
class ChatGPTSQLToRELResponseIncremental(Base):
    __tablename__ = 'chatgpt_sql_to_rel_incremental'
    kaggle_ref_incremental = Column(String, primary_key=True)
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

class RelExecutionRerunIncremental(Base):
    __tablename__ = 'rel_execution_rerun_incremental'
    ref_incremental = Column(String, primary_key=True)
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

    # def export_table(self):
    #     # if not table_id.startswith('table'):
    #     #     table_id = 'table_{}'.format(table_id.replace('-', '_'))
    #     table_info = \
    #         self.conn.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[0].as_dict()[
    #             "sql"]
    #     schema_str = schema_re.findall(table_info)[0]
    #     schema = {}
    #     for tup in schema_str.split(', '):
    #         c, t = tup.split()
    #         schema[c] = t
    #
    #     query = 'SELECT * FROM {}'.format(table_id)
    #     out = self.conn.query(query)
    #     return schema, out.as_dict()


import openai


class MyException(Exception):
    pass


class SchemaToSQLChatGPT(ChatGPT):
    def __init__(self, subset_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subset_path = subset_path
        self.api_call_cache_db = self.connect_sqlite_api_call_cache_db()

        self.leading_number_pattern = re.compile("^\d+.")
        self.double_newline_pattern = re.compile("\n *\n")
        self.code_block_pattern = re.compile("```\s*\d+.")
        self.match_one_dot_pattern = re.compile("1\.")
        self.match_desc_code_pattern = re.compile("[\s\S]+```[\s\S]+```")
        self.sql_per_ds = 15

    def sql_synth_prompt(self, one_schema, one_top_row, simple=False):
        num_tables = len(one_schema)
        instruction = f"""
We have a SQL database with {num_tables} tables. For each table, here is the schema and a sample row data.
"""
        for idx, (schema, row) in enumerate(zip(one_schema, one_top_row)):
            instruction += f"""
Schema for table {idx}:
{schema}
A sample row data from table {idx}:
{row}
"""
        use_joins = "If possible, use joins." if len(one_schema) > 1 else ""
        simple = " with relatively high complexity" if not simple else ""

        instruction += f"""
Consider the typical users who access this database. What kind of SQLite SELECT queries would they write? List {self.sql_per_ds} examples SQL code that are directly executable{simple}, each following a description.
{use_joins} Reply in the format of a description and the SQL code for each example
"""
        return instruction

    def split_by_code_block(self, answer):
        s = re.split(self.code_block_pattern, answer)
        if len(s) == 1:
            raise RuntimeError
        res = self.match_one_dot_pattern.split(s[0])
        if len(res) > 1:
            res = res[1]
        else:
            res = res[0]
        s[0] = res
        start, end = re.match(self.match_desc_code_pattern, s[-1]).regs[0]
        s[-1] = s[-1][start: end]
        return s

    def split_by_double_new_line(self, answer):
        s = re.split(self.double_newline_pattern, answer)
        if len(s) > self.sql_per_ds:
            if "1." in s[0]:
                start = 0
            else:
                start = 1
            s = s[start:]
            if "SELECT" not in s[-1]:
                s = s[:-1]
        return s

    def process_chat_gpt_answers(self, answer):
        try:
            s = self.split_by_code_block(answer)
        except RuntimeError:
            s = self.split_by_double_new_line(answer)
        desc_sql = []
        descs = []
        sqls = []
        if len(s) == self.sql_per_ds * 2:
            descs = s[0::2]
            sqls = s[1::2]
        else:
            for da in s:
                splitted = da.split('\n')
                desc = splitted[0]
                sql = splitted[1:]
                sql = "\n".join(sql)
                descs.append(desc)
                sqls.append(sql)
        try:
            for desc, sql in zip(descs, sqls):
                sql = sql.replace("SQL:", "")
                sql = sql.replace("SQL", "")
                sql = sql.replace("code:", "")
                sql = sql.replace("Code:", "")

                desc = desc.strip()
                sql = sql.strip()
                # assert "SELECT" not in desc
                assert "SELECT" in sql
                # remove numbering
                m = self.leading_number_pattern.match(desc)
                if m is not None:
                    desc = desc[m.regs[0][1]:]
                    desc = desc.strip()
                    desc = desc.strip(":")
                sql = sql.strip('```')
                sql = sql.strip('sql')
                sql = sql.strip()
                desc_sql.append((desc, sql))
        except AssertionError:
            desc_sql = None

        return desc_sql

    async def synthesize_sql_code(self, one_row, one_schema, one_top_row, verbose_level=1):
        """
        More efficient synthesis that reuses the prompt for multiple generations
        :param batch:
        :return:
        """
        chatgpt = ChatGPT()
        synths = []
        instruction = self.sql_synth_prompt(one_schema, one_top_row)

        if verbose_level > 1:
            print(instruction)

        answer = await chatgpt.query_chat_gpt_async(instruction, model="gpt-3.5-turbo-16k")
        splitted_answers = self.process_chat_gpt_answers(answer)
        return splitted_answers

    def connect_sqlite_api_call_cache_db(self):
        engine = create_engine(f"sqlite:///{kaggle_path}/openai_sql_cache.db")
        # Create the table in the database
        Base.metadata.create_all(engine)

        # Create a session to interact with the database
        Session = sessionmaker(bind=engine)
        session = scoped_session(Session)
        # session = Session()
        try:
            _ = session.connection()
        except PendingRollbackError:
            session.rollback()
        return session

    def load_api_call_cache_db(self, ref):
        res = self.api_call_cache_db.query(ChatGPTSQLResponse).filter_by(kaggle_ref=ref).first()
        if res is None:
            return None
        else:
            res = json.loads(res.openai_response)
            return res

    def write_api_call_cache_db(self, ref, res):
        exist = self.api_call_cache_db.query(ChatGPTSQLResponse).filter_by(kaggle_ref=ref).first()
        if exist:
            self.rm_row_api_all_cache_db(ref)
        res_str = json.dumps(res)
        new_response = ChatGPTSQLResponse(kaggle_ref=ref, openai_response=res_str)
        self.api_call_cache_db.add(new_response)
        self.api_call_cache_db.commit()

    def rm_row_api_all_cache_db(self, ref):
        res = self.api_call_cache_db.query(ChatGPTSQLResponse).filter_by(kaggle_ref=ref).first()
        self.api_call_cache_db.delete(res)
        self.api_call_cache_db.commit()

    async def async_synthesize_sql_code_with_retry(self, one_row, one_schema, one_top_row, semaphore, progress):
        async with semaphore:
            retry = 5
            while retry > 0:
                try:
                    if one_schema is None:
                        return None
                    else:
                        ref = one_row['ref']
                        cache = self.load_api_call_cache_db(ref)
                        if cache is not None and len(cache) == self.sql_per_ds:
                            progress.update(1)
                            return cache
                        else:
                            res = await self.synthesize_sql_code(one_row, one_schema, one_top_row)
                            # if any(["table_name" in r[1] for r in res]):
                            #     raise MyException
                            if res is not None and len(res) == self.sql_per_ds:
                                self.write_api_call_cache_db(ref, res)
                            progress.update(1)
                            return res
                except (openai.error.InvalidRequestError,) as e:
                    progress.update(1)
                    return None
                except openai.error.OpenAIError as e:
                    time.sleep(10)
                    retry -= 1
                except MyException:
                    raise MyException
                except (Exception,) as e:
                    progress.update(1)
                    return None
            return None

    async def synthesize_all_sql_code_async(self, catalog, schemas, top_rows, max_in_flight=10):
        # asynchronous semaphore ping API
        semaphore = asyncio.Semaphore(max_in_flight)
        tasks = []
        progress = tqdm(total=len(schemas), desc="Synthesizing SQL code")
        for one_row, one_schema, one_top_row in zip(catalog.iloc, schemas, top_rows):
            task = self.async_synthesize_sql_code_with_retry(one_row, one_schema, one_top_row, semaphore, progress)
            tasks.append(task)
        sql_synths = await asyncio.gather(*tasks)
        return sql_synths

    def synthesize_all_sql_code(self, catalog, schemas, top_rows):
        # synchronous entry
        sql_synths = asyncio.run(self.synthesize_all_sql_code_async(catalog, schemas, top_rows))
        return sql_synths


class KagglePrompting(WikiSQLPrompting):
    def __init__(self):
        self.cache_path = project_root() / "cache/kaggle"

    @staticmethod
    def manual_examples(number_of_manual_examples = None):
        queries = [
            {
                "ref": 'iamsouravbanerjee/data-science-salaries-2023',
                "sql": "SELECT L.JobTitle, L.ExperienceLevel, V3.JobTitle, V3.ExperienceLevel "
                       "FROM LatestDataScienceSalaries AS L "
                       "INNER JOIN V3LatestDataScienceSalaries AS V3 ON L.CompanyLocation = V3.CompanyLocation "
                       'WHERE L.CompanyLocation = "United States";',
                "rel": 'def temp[x,y] = LatestDataScienceSalaries:CompanyLocation[x, company_location] and V3LatestDataScienceSalaries:CompanyLocation[y, company_location] and company_location = "United States" from company_location\n'
                       'def output[x_job_title, x_experience_level, y_job_title, y_experience_level]  = LatestDataScienceSalaries:JobTitle[x, x_job_title],  LatestDataScienceSalaries:ExperienceLevel[x, x_experience_level], V3LatestDataScienceSalaries:JobTitle[y, y_job_title], V3LatestDataScienceSalaries:ExperienceLevel[y, y_experience_level] and temp[x,y] for x, y ',
            }, {
                "ref": 'bhanupratapbiswas/instgram',
                "sql": 'SELECT p.ImageLink '
                       'FROM Photos p '
                       'JOIN Likes l ON p.Id = l.Photo '
                       'JOIN Users u ON l.User = u.Id '
                       'LEFT JOIN Follows f ON u.Id = f.Follower AND f.Followee = p.UserID '
                       'WHERE f.Follower IS NULL;',
                "rel": """def UserPhoto(user_id, photo_id) = Photos:UserID[l,user_id] and Photos:Id[l, photo_id] from l
def UserLikesPhoto(user_id, photo_id) = Likes:User[l,user_id] and Likes:Photo[l, photo_id] from l
def UserID(user_id) = Users:Id(l, user_id) from l
def UserFollowsUser(follower_id, followee_id) = Follows:Follower[l,follower_id] and Follows:Followee[l, followee_id] from l
def PhotoIDHasImageLink(photo_id, image_link) =
    Photos:Id(p, photo_id) and Photos:ImageLink(p, image_link) from p
def photos_liked_by_not_a_follower(photo) =
    UserPhoto(poster, photo)
    and UserLikesPhoto(liker, photo)
    and not UserFollowsUser(liker, poster)
    from poster, liker
def output = PhotoIDHasImageLink[photos_liked_by_not_a_follower]""",
            }, {
                "ref": 'konradb/us-military-interventions',
                "sql": 'SELECT MainTable.StateB, MainTable.Name FROM MainTable '
                       'JOIN CaseUniverse ON MainTable.StateB = CaseUniverse.TargetState '
                       'AND MainTable.StateBCode = CaseUniverse.TargetCOWID '
                       'WHERE TargetState = "FRN" AND TargetCOWID = "220";',
                "rel": 'def join[mt, cu] = MainTable:StateB[mt, sb] and CaseUniverse:TargetState[cu, sb] and MainTable:StateBCode[mt, sbc] and CaseUniverse:TargetCOWID[cu, sbc] and sb="FRN" and sbc="220" from sb, sbc \n'
                       'def output = MainTable:StateB[mt], MainTable:Name[mt] for mt, cu where join[mt, cu]',
            }, {
                "ref": 'twinkle0705/state-wise-power-consumption-in-india',
                "sql": "SELECT LD.Regions, SUM(DT.Punjab + DT.Haryana + DT.Rajasthan + DT.Delhi + DT.UP + DT.Uttarakhand) AS Total_Usage "
                       "FROM DatasetTk DT "
                       "INNER JOIN LongData LD ON DT.Punjab = LD.Usage AND LD.States = 'Punjab' "
                       "GROUP BY LD.Regions; ",
                "rel": """def join1[dt,ld]= DatasetTk:Punjab[dt, punjab] and LongData:Usage[ld, punjab] from punjab

def sum_usage = DatasetTk:Punjab[dt] + DatasetTk:Haryana[dt] +
DatasetTk:Rajasthan[dt] + DatasetTk:Delhi[dt] +
DatasetTk:UP[dt] + DatasetTk:Uttarakhand[dt]  for dt

def output[region] = sum[dt, usage: 
LongData:Regions[ld, region] 
and sum_usage[dt, usage]
and join1[dt,ld] and LongData:States[ld] = "Punjab" for ld]
                """,
            }, {
                "ref": 'vetrirah/customer',
                "sql": 'SELECT Test.ID, Test.Age, Train.SpendingScore '
                       'FROM Test '
                       'INNER JOIN Train ON Test.ID = Train.ID '
                       'WHERE Test.Age >= 40 OR Train.Age >= 40; ',
                "rel": 'def output[id, age, spending_score] = Test:ID[test_idx, id], Test:Age[test_idx, age], Train:SpendingScore[train_idx, spending_score] and '
                       ' Test:ID[test_idx, id] and Train:ID[train_idx, id] and (Test:Age[test_idx]>=40 or Train:Age[train_idx] >= 40) from test_idx, train_idx',
            }, {
                "ref": 'vora1011/ipl-2022-match-dataset',
                "sql": 'SELECT bb.PlayerOut, m.WinningTeam '
                       'FROM IPLBallByBall2022 bb '
                       'JOIN IPLMatches2022 m ON bb.ID = m.ID '
                       'WHERE bb.ID = 1312200 AND bb.IsWicketDelivery = 1; ',
                "rel": 'def join[bb, m] = IPLBallByBall2022:ID[bb, id] and IPLMatches2022:ID[m, id] and IPLBallByBall2022:ID[bb]=1312200 and IPLBallByBall2022:IsWicketDelivery[bb]=1 from id \n'
                       'def output = IPLBallByBall2022:PlayerOut[bb], IPLMatches2022:WinningTeam[m] from bb, m where join[bb,m] ',
            }, {
                "ref": 'nenriki/real-instagram-accounts-kz',
                "sql": 'SELECT il.AccountName, il.Ismarketplace '
                       'FROM InstAccLabeled AS il '
                       'JOIN InstAcc AS ia ON il.AccountName = ia.Username '
                       'WHERE ia.IsBusinessAccount = 1;',
                "rel": 'def join[il, ia] = InstAccLabeled:AccountName[il, account_name] and InstAcc:Username[ia, account_name] and InstAcc:IsBusinessAccount[ia]=boolean_true from account_name \n'
                       'def output = InstAccLabeled:AccountName[il], InstAccLabeled:Ismarketplace[il]  for il, ia where join[il, ia]',
            },
            # good but too complicated?
            {
                "ref": 'justinas/housing-in-london',  # 93
                "sql": 'SELECT table0.Area, table0.AveragePrice, table1.MedianSalary '
                       'FROM HousingInLondonMonthlyVariables AS table0 '
                       'JOIN HousingInLondonYearlyVariables AS table1 '
                       'ON table0.Code = table1.Code '
                       'WHERE table0.BoroughFlag = 1;',
                "rel": 'def join[row_id_0, row_id_1] = HousingInLondonMonthlyVariables:Code[row_id_0, code] and HousingInLondonYearlyVariables:Code[row_id_1, code] and HousingInLondonMonthlyVariables:BoroughFlag[row_id_0]=1 from code   \n'
                       'def output[area, average_price, median_salary] = HousingInLondonMonthlyVariables:Area[row_id_0, area] and HousingInLondonMonthlyVariables:AveragePrice[row_id_0, average_price] and HousingInLondonYearlyVariables:MedianSalary[row_id_1, median_salary] and join[row_id_0, row_id_1] for row_id_0, row_id_1  ',
            },
            {
                "ref": 'kaushiksuresh147/customer-segmentation',
                "sql": "SELECT Gender, Segmentation, COUNT(*) FROM Test WHERE Segmentation IN ('B', 'C') "
                       "GROUP BY Gender, Segmentation;",
                "rel": 'def output[gender, seg] = count[row_id: Test:Gender(row_id, gender) and Test:Segmentation(row_id,seg) and {"B"; "C"}(seg)] ',
            },
            {
                "ref": 'adelanseur/leading-causes-of-death-in-the-us-1999-2017',
                "sql": 'SELECT DISTINCT State FROM NCHSLeadingCausesOfDeathUnitedStates; ',
                "rel": 'def output[state] = NCHSLeadingCausesOfDeathUnitedStates:State[x,state] from x',
            }, {
                "ref": "mfaisalqureshi/hr-analytics-and-job-prediction",
                "sql": "SELECT * FROM HRCommaSep;",
                "rel": "def output[col, row, val] = HRCommaSep[col, row, val]"
            }, {
                "ref": "ruchi798/tv-shows-on-netflix-prime-video-hulu-and-disney",
                "sql": "SELECT Title, Year FROM TvShows;",
                "rel": "def output[title, year] = TvShows:Title[idx, title], TvShows:Year[idx, year] for idx"
            }, {
                "ref": "mfaisalqureshi/hr-analytics-and-job-prediction",
                "sql": "SELECT AverageMontlyHours FROM HRCommaSep WHERE Department = 'sales';",
                "rel": 'def output[average_monthly_hours] = HRCommaSep:Department(row_idx, "sales") and HRCommaSep:AverageMontlyHours(row_idx, average_monthly_hours) for row_idx'
            }, {
                "ref": "aprabowo/indonesia-tourism-destination",
                "sql": "SELECT Location, Age FROM User WHERE UserId = 1;",
                "rel": "def output[location, age] = User:Location[idx, location], User:Age[idx, age], User:UserId[idx, 1] for idx "
            }, {
                "ref": "crxxom/chess-gm-players",
                "sql": "SELECT * FROM GMPlayersStatistics WHERE isstreamer = 1 AND bulletwin > 70;",
                "rel": """def output[col, row, val] = GMPlayersStatistics[col, row, val] and GMPlayersStatistics:IsStreamer[row] = boolean_true and GMPlayersStatistics:BulletWin[row] > 70  """
            }, {  # THIS ONE APPEARS TO BE EMPTY?
                "ref": "bhanupratapbiswas/weather-data",
                "sql": 'SELECT AVG(Visibilitykm), MAX(WindSpeedkmh) FROM WeatherData WHERE Weather = "Rain";',
                "rel": 'def all_row_ids_with_rain[row_id] = WeatherData:Weather[row_id]="Rain" \n'
                       'def output = average[row_id, value_vis: all_row_ids_with_rain(row_id) and WeatherData:VisibilityKm(row_id, value_vis)], max[row_id, value_wind_speed : all_row_ids_with_rain(row_id) and WeatherData:WindSpeedKmh(row_id, value_wind_speed)] '
            }, {
                "ref": "geomack/spotifyclassification",
                "sql": "SELECT * FROM Data WHERE Danceability > 0.8;",
                "rel": "def output[col, idx, val] = Data[col, idx, val] and Data:Danceability[idx] > 0.8"
            }, {
                "ref": "agirlcoding/all-space-missions-from-1957",
                "sql": "SELECT * FROM SpaceCorrected WHERE StatusMission IN ('Success', 'Failure');",
                "rel": 'def output[col, row_id, val] = SpaceCorrected[col, row_id, val] and {"Success"; "Failure"}(SpaceCorrected:StatusMission[row_id])'
            }, {
                "ref": "agirlcoding/all-space-missions-from-1957",
                "sql": "SELECT * FROM SpaceCorrected WHERE Location LIKE '%Florida%';",
                "rel": 'def output[col, row_id, val] = SpaceCorrected[col, row_id, val] and like_match("\%Florida\%", SpaceCorrected:Location[row_id])'
            }, {
                "ref": "ujjwalwadhwa/cars24com-used-cars-dataset",
                "sql": "SELECT * FROM Cars24Combined WHERE Fuel = 'PETROL';",
                "rel": 'def output[colname, row_idx, val] = Cars24Combined[colname, row_idx, val] and Cars24Combined[:Fuel, row_idx, "PETROL"]'
            }, {
                "ref": "khushipitroda/uci-dataset",
                "sql": "SELECT DISTINCT Name FROM UCIDatasets;",
                "rel": "def output[name] = UCIDatasets:Name[idx, name] from idx "
            }, {
                "ref": "aungpyaeap/supermarket-sales",
                "sql": "SELECT COUNT(*) FROM SupermarketSalesSheet1;",
                "rel": "def count_rows = count[row_id: SupermarketSalesSheet1[_, row_id, _]]\n "
                       "def output = count_rows"
            }, {
                "ref": "khushipitroda/internshala-jobs-dataset-with-5000-rows",
                "sql": 'SELECT * FROM Jobs WHERE ActivelyHiring = 1.0;',
                "rel": "def output[col, idx, val] = Jobs[col, idx, val] and Jobs:ActivelyHiring[idx] = 1.0"
            }, {
                "ref": 'mauryansshivam/marvel-comics-1500-characters-and-their-appearance',
                "sql": "SELECT marvelcomiccharactername, comicappearance1 FROM MarvelComicsLegacy WHERE marvelcomiccharactername = 'Wolverine';",
                "rel": 'def output[marvel_character_name, comic_appearance_1] = MarvelComicsLegacy:MarvelComicCharacterName[idx, marvel_character_name], MarvelComicsLegacy:ComicAppearance1[idx, comic_appearance_1] and marvel_character_name = "Wolverine" for idx'
            }, {
                "ref": 'blastchar/telco-customer-churn',
                "sql": 'SELECT * FROM WAFnUseCTelcoCustomerChurn WHERE Churn = "Yes";',
                "rel": 'def output[col, row_idx, val] = WAFnUseCTelcoCustomerChurn[col, row_idx, val] and WAFnUseCTelcoCustomerChurn:Churn[row_idx] = "Yes"'
            }, {
                "ref": 'alexisbcook/data-for-datavis',
                "sql": 'SELECT AVG(Charges) FROM Insurance WHERE Age > 40;',
                "rel": 'def output = average[idx, charges: Insurance:Charges[idx, charges] and Insurance:Age[idx] > 40]'
            }, {
                "ref": 'sanjanchaudhari/user-behavior-on-instagram',
                "sql": 'SELECT * FROM CommentsCleaned WHERE EmojiUsed = "yes";',
                "rel": 'def output[col, row_id, val] = CommentsCleaned[col, row_id, val] and CommentsCleaned:EmojiUsed[row_id] = "yes"'
            }, {
                "ref": "ujjwalwadhwa/cars24com-used-cars-dataset",
                "sql": 'SELECT * FROM Cars24Combined WHERE Fuel = "PETROL" AND Location = "HR-98";',
                "rel": 'def output[col, idx, val] = Cars24Combined[col, idx, val] and Cars24Combined:Fuel[idx, "PETROL"] and Cars24Combined:Location[idx, "HR-98"]'
            }, {
                "ref": "bhanupratapbiswas/national-family-health-survey-nfhs-2019-21",
                "sql": 'SELECT StateUT, AVG(NumberOfHouseholdsSurveyed) as AverageHouseholds '
                       'FROM Datafile '
                       'GROUP BY StateUT;',
                "rel": 'def output[state] = average[idx, num: Datafile:NumberOfHouseholdsSurveyed[idx, num] and Datafile:StateUT[idx, state]]'
            }, {
                "ref": "ujjwalwadhwa/cars24com-used-cars-dataset",
                "sql": "SELECT * FROM Cars24Combined WHERE Fuel = 'PETROL' AND Location = 'HR-98';",
                "rel": 'def output[col, idx, val] = Cars24Combined[col, idx, val] and Cars24Combined:Fuel[idx, "PETROL"] and Cars24Combined:Location[idx, "HR-98"]'
            }, {
                "ref": 'pkdarabi/diabetes-dataset-with-18-features',
                "sql": "SELECT Gender, COUNT(*) FROM Diabetes GROUP BY Gender;",
                "rel": 'def output[gender] = count[row_id: Diabetes:Gender[row_id, gender]]'
            }, {
                "ref": 'whenamancodes/data-professionals-salary-dataset-2022',
                "sql": "SELECT JobTitle, AVG(Salary) AS AverageSalary\nFROM PartiallyCleanedSalaryDataset\nGROUP BY JobTitle;",
                "rel": "def output[job_title] = average[idx, salary: PartiallyCleanedSalaryDataset:Salary[idx, salary] and PartiallyCleanedSalaryDataset:JobTitle[idx, job_title]]"
            }, {
                "ref": 'jtrotman/formula-1-pitstops-1994-2010',
                "sql": 'SELECT Race, COUNT(*) AS TotalPitstops\nFROM Pitstops\nGROUP BY Race;',
                "rel": "def output[race] = count[row_id: Pitstops:Race[row_id, race]]"
            }, {
                "ref": 'shreyapurohit/anime-data',
                "sql": 'SELECT Title, Rating FROM TopAnime WHERE Rating = (SELECT MAX(Rating) FROM TopAnime);',
                "rel": "def max_rating = max[x, rating : TopAnime:Rating[x, rating]] \n"
                       "def output[title, rating] = TopAnime:Title[idx, title], TopAnime:Rating[idx, rating] and TopAnime:Rating[idx, max_rating] for idx"
            }, {
                "ref": 'bravehart101/sample-supermarket-dataset',
                "sql": 'SELECT Category, SUM(Sales) AS TotalSales, SUM(Profit) AS TotalProfit \nFROM SampleSuperstore \nGROUP BY Category;',
                "rel": "def output[category] = sum[row_id, sales: SampleSuperstore:Category(row_id, category) and SampleSuperstore:Sales(row_id, sales)], sum[row_id, profit: SampleSuperstore:Category(row_id, category) and SampleSuperstore:Profit(row_id, profit)]"
            }, {
                "ref": "poojakeer/e-commerce-dataset",
                "sql": "SELECT SUM(CustomerCareCalls) \nFROM Train \nWHERE DiscountOffered >= 50;",
                "rel": "def output = sum[row_id,calls: Train:CustomerCareCalls[row_id, calls] and Train:DiscountOffered[row_id] >= 50]"
            }, {
                "ref": 'vikramamin/customer-churn-decision-tree-and-random-forest',
                "sql": "SELECT AVG(MonthlyCharges) FROM CustomerChurn WHERE Churn = 'Yes';",
                "rel": 'def output = average[row_id, monthly_charges: CustomerChurn:MonthlyCharges(row_id, monthly_charges) and CustomerChurn:Churn[row_id] = "Yes"]'
            }, {
                "ref": 'ashishg21/facebook-live-sellers-in-thailand-uci-ml-repo',
                "sql": 'SELECT COUNT(DISTINCT StatusType) AS UniqueStatusTypes\nFROM Live;',
                "rel": 'def output = count[status_type: Live:StatusType(row_id, status_type) from row_id]'
            }, {
                "ref": 'prasoonkottarathil/polycystic-ovary-syndrome-pcos',
                "sql": 'SELECT * FROM PCOSInfertility ORDER BY SlNo DESC;',
                "rel": 'def output[col, row_idx, val] = PCOSInfertility[col, row_idx, val]'
            }, {
                "ref": 'rutuspatel/walmart-dataset-retail',
                "sql": 'SELECT SUM(WeeklySales) AS TotalWeeklySales FROM WalmartStoreSales;',
                "rel": "def output = sum[row_idx, weekly_sales : WalmartStoreSales:WeeklySales[row_idx, weekly_sales]]"
            }, {
                "ref": 'farukalam/weather-timeseries-data-in-sylhet-bangladesh',
                "sql": 'SELECT * FROM WeatherData ORDER BY RelativeHumidity ASC;',
                "rel": 'def output[col, row_id, val] = WeatherData[col, row_id, val]'
            }, {
                "ref": 'roopacalistus/superstore',
                "sql": 'SELECT ShipMode, SUM(Sales) AS TotalSales FROM SampleSuperstore GROUP BY ShipMode;',
                "rel": 'def output[shipmode] = sum[row_id, value_sales: SampleSuperstore:Sales[row_id, value_sales] and SampleSuperstore:ShipMode[row_id, shipmode]]'
            }, {
                "ref": 'vora1011/ipl-2022-match-dataset',
                "sql": 'SELECT PlayerOfMatch, WinningTeam\nFROM IPLMatches2022\nWHERE WinningTeam = "Rajasthan Royals";',
                "rel": 'def output[player_of_match, winning_team] = IPLMatches2022:PlayerOfMatch[idx, player_of_match], IPLMatches2022:WinningTeam[idx, winning_team] and IPLMatches2022:WinningTeam[idx, winning_team] and winning_team = "Rajasthan Royals" for idx'
            }, {
                "ref": 'niharika41298/gym-exercise-data',
                "sql": 'SELECT Title, Desc FROM MegaGymDataset WHERE Type = "Strength";',
                "rel": 'def output[title, desc] = MegaGymDataset:Title[idx, title], MegaGymDataset:Desc[idx, desc]  and MegaGymDataset:Type[idx, "Strength"] for idx'
            }, {
                "ref": 'khushipitroda/internshala-internship-dataset',
                "sql": "SELECT TypeOfInternship, COUNT(*) as Count FROM Internship GROUP BY TypeOfInternship;",
                "rel": 'def output[type_of_internship] = count[row_id: Internship:TypeOfInternship[row_id, type_of_internship]]'
            }
        ]
        if number_of_manual_examples is not None:
            ret =  queries[:number_of_manual_examples]
            return ret
        else:
            return queries

    def failed_examples(self):
        e = [{
            "ref": "ujjwalwadhwa/cars24com-used-cars-dataset",
            "sql": "SELECT CarName, Year, Price FROM Cars24Combined ORDER BY Price DESC;",
            "rel": ""
        }, {
            "ref": '',
            "sql": '',
            "rel": ''
        }, {
            "ref": '',
            "sql": '',
            "rel": ''
        }, {
            "ref": '',
            "sql": '',
            "rel": ''
        }, ]
        return e

    def make_prototype_prompt(self, number_of_manual_examples=None):
        instruction = f"""
REL is a database management system language that is similar to datalog. REL is based on the sixth normal form (6NF),
where every variable represents a relation, i.e. a set of tuples.
In REL, from-variables will not appear in the output, and for-variables will appear in the output.
Strings in REL use double quotes.
Below are few examples of SQL code and REL code pairs that perform the same query.

Examples:
    """
        for i, ex in enumerate(self.manual_examples(number_of_manual_examples=number_of_manual_examples)):
            sql, rel = ex['sql'], ex['rel']
            instruction += f"""
{i + 1}.SQL: 
```
{sql}
```
"""
            instruction += f"""
{i + 1}.REL: 
```
{rel}
```
"""
        return instruction


def upload_to_rai(row, iloc, synth_pipeline, relx):
    name = row['ref']
    csv_files = synth_pipeline.kaggle.find_files(name)[0]

    number_unique_name_to_readable_name = {}
    readable_name_to_number_unique_name = {}
    schemas = []
    for csv_idx, csv_file in enumerate(csv_files):
        tbl_name = csv_file.stem
        tbl_name = synth_pipeline.sql.table_name_sanitize(tbl_name)
        number_unique_name = f"ds_{iloc}_tbl_{csv_idx}"

        df = pandas.read_csv(csv_file)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        df = df.rename(columns=synth_pipeline.sql.rename_columns(df.columns))
        df = df.iloc[:SQLMagic.max_rows]
        csv_string = io.StringIO()
        df.to_csv(csv_string)
        csv_string = csv_string.getvalue()
        schema = synth_pipeline.get_schema_from_sql(iloc, tbl_name)
        schemas.append(schema)
        # relx.delete_relation(number_unique_name)
        relx.upload_csv(csv_string, number_unique_name, schema)
        number_unique_name_to_readable_name[number_unique_name] = tbl_name
        readable_name_to_number_unique_name[tbl_name] = number_unique_name

    # execute rel query
    table_loading_code = ""
    for number_unique_name, readable_name in number_unique_name_to_readable_name.items():
        table_loading_code += f"""
def {number_unique_name}_cols = {number_unique_name}[col, _, _] for col
def {number_unique_name}_rows = {number_unique_name}[_, row, _] for row
def {number_unique_name}_xprod = {number_unique_name}_cols, {number_unique_name}_rows
def {number_unique_name}_existing = {number_unique_name}[col, row, _] for col, row

def {number_unique_name}_toadd = diff[{number_unique_name}_xprod, {number_unique_name}_existing], missing

def {readable_name} = union[{number_unique_name}, {number_unique_name}_toadd]
        """

    return table_loading_code, number_unique_name_to_readable_name, readable_name_to_number_unique_name, schemas


def post_process(rel_code):
    r = rel_code.replace("value", "value1")
    r = r.replace("'", '"')
    return r


def run_rel_rai(rels, table_loading_code, relx, print_error=False):
    responses = []
    codes = []
    rel_times = []
    for rel_code in rels:
        rel_code = post_process(rel_code)
        code = f"""
{table_loading_code}
{rel_code}
        """
        codes.append(code)
        tic = time.time()
        rsp = relx.execute_query(code, print_error=print_error)
        rel_times.append(time.time() - tic)
        responses.append(rsp)
    return codes, responses, rel_times


def upload_and_run_thread_function(row, iloc, sqls, same_ds_synth, ready_synth_pipeline_queue, res,
                                   write_to_cache=True):
    # print("4")
    pipeline = ready_synth_pipeline_queue.get(block=True)
    # res = pipeline.find_rel_execution_results(row['ref'])
    # print("3")
    try:
        if res is None:
            sql_magic = pipeline.make_sql(row, f"/{pipeline.relx_identifier}:memory:")
            # f"file:memdb{pipeline.relx_identifier}?mode=memory")
            rels = []
            sql_results = []
            submitted_sqls = []
            sql_times = []
            for synth in same_ds_synth:
                _, sql_idx = synth['idx']
                desc, sql = sqls[sql_idx]
                tic = time.time()
                sql_result = sql_magic.execute_query(sql)
                sql_time = time.time() - tic
                sql_times.append(sql_time)
                sql_results.append(sql_result)
                rels.append(synth['query'])
                submitted_sqls.append((desc, sql))
            sql_magic.remove_all_tables_terminate()
            table_loading_code, number_unique_name_to_readable_name, readable_name_to_number_unique_name, schemas = \
                upload_to_rai(row, iloc, pipeline, pipeline.relx)
            codes, responses, rel_times = run_rel_rai(rels, table_loading_code, pipeline.relx, print_error=False)

            acc, infos, problems = pipeline.compare_sql_rel_results(responses, sql_results, rels, submitted_sqls, codes,
                                                                    row=row, sql_magic=sql_magic)
            if write_to_cache:
                pipeline.write_rel_execution_results(row['ref'], acc, infos, sql_times, rel_times, problems)
        else:
            acc, infos = res
        return row['ref'], acc, infos
    except RuntimeError:
        # major issues are wrapped as RuntimeError, such as authentication problem
        traceback.print_exc()
        raise
    except HTTPError:
        traceback.print_exc()
        if write_to_cache:
            pipeline.write_rel_execution_results(row['ref'], None, None, None, None, None)
        raise
    except:
        traceback.print_exc()
        if write_to_cache:
            pipeline.write_rel_execution_results(row['ref'], None, None, None, None, None)
            # TODO potentially not the entire batch timed out
        return None, None, None
    finally:
        ready_synth_pipeline_queue.put(pipeline)


def worker_translate_sql_to_rel(pipeline, chatgpt, cache, row, batch, ready_queue, pbar):
    worker_session = ready_queue.get(block=True)
    retry = 5
    db_key = row['ref'] if pipeline.incremental_manual_examples is None else row['ref']+str(pipeline.incremental_manual_examples)
    while retry > 0:
        try:
            if cache is None:
                try:
                    raw_answer, synths = chatgpt.proto_synthesis_one_batch(batch, verbose_level=0, incremental_manual_examples = pipeline.incremental_manual_examples)
                except RuntimeError:
                    return None, None
                pipeline.write_sql_to_rel_cache_db(db_key, (batch, raw_answer, synths), worker_session)
            else:
                last_batch, raw_answer, synths = cache
                last_batch_idx = tuple(tuple(l['idx']) for l in last_batch)
                batch_id = tuple(tuple(l['idx']) for l in batch)
                batch_identical = True
                for i in range(len(batch)):
                    if last_batch[i]['query'] != batch[i]['query']:
                        batch_identical = False
                if batch_identical:
                    if last_batch_idx != batch_id:
                        pipeline.rm_sql_to_rel_cache_db(db_key)
                        pipeline.write_sql_to_rel_cache_db(db_key, (batch, raw_answer, synths), worker_session)
                    else:
                        # no need to modify
                        pass
                else:
                    raw_answer, synths = chatgpt.proto_synthesis_one_batch(batch, verbose_level=0)
                    pipeline.rm_sql_to_rel_cache_db(db_key)
                    pipeline.write_sql_to_rel_cache_db(db_key, (batch, raw_answer, synths), worker_session)
            pbar.update(1)
            return raw_answer, synths
        except (openai.error.RateLimitError, openai.error.APIError) as e:
            retry -= 1
            if retry == 0:
                pbar.update(1)
                return None, None
            else:
                time.sleep(20)
        finally:
            ready_queue.put(worker_session)
    pbar.update(1)
    return None, None


class SynthesisPipeline:
    """
    # 0
    self.kaggle.catalog_datasets()
    # 1
    self.kaggle.download_datasets(step)
    # 2
    self.get_all_schemas_and_top_rows(step)
    # 3
    self.chatgpt_synthesize_sql_queries(step)
    # 4
    self.filter_sql_execution_with_ground_truth(step)
    # 5, skipped unless run directly in dev mode
    self.develop_sql_rel_examples(step)
    # 6
    self.chatgpt_translate_sql_to_rel(step)
    # 7
    """

    def __init__(self, dev=False, relx_identifier="", incremental_manual_examples=None, make_engine=True):
        self.incremental_manual_examples = incremental_manual_examples
        self.dev = dev
        self.dev_slice = 100
        self.kaggle = KaggleDownloader()
        self.chatgpt = SchemaToSQLChatGPT(subset_path=self.kaggle.subset_path())
        self.sql = SQLMagic()
        self.relx_identifier = relx_identifier
        self.relx = RELX(verbose=False, postfix=relx_identifier)
        if relx_identifier != "":
            if make_engine:
                self.relx.make_database()
                self.relx.make_engine()
        dev_st = "_dev2" if self.dev else ""
        self.dev_st = dev_st
        self.schema_path = kaggle_path / f"schema{dev_st}.json"
        self.schemas = []
        self.raw_schemas = []
        self.top_row_path = kaggle_path / f"top_row{dev_st}.json"
        self.synthesized_sql_path = kaggle_path / f"synthesized_sql{dev_st}.json"
        self.synthesized_sql = None
        self.executable_flags_path = kaggle_path / f"executable_flags{dev_st}.json"
        self.sql_executable_flag = None
        # self.sql_to_rel_translated_path = kaggle_path / f"sql_to_rel_translated{dev_st}.json"
        self.sql_to_rel_translated = None
        self.sql_to_rel_translated_raw_answers = None
        # self.rel_execution_accuracy_path = kaggle_path / f"rel_execution_accuracy{dev_st}_rerun.json"
        self.rel_execution_accuracy = None
        self.sql_to_rel_cache_db = self.connect_sql_to_rel_cache_db()
        self.execution_flags = None

    def connect_sql_to_rel_cache_db(self):
        engine = create_engine(f"sqlite:///{kaggle_path}/openai_sql_to_rel_cache_db.db?check_same_thread=False")
        # Create the table in the database
        Base.metadata.create_all(engine)

        # Create a session to interact with the database
        Session = sessionmaker(bind=engine)
        session = scoped_session(Session)
        # session = Session()
        try:
            _ = session.connection()
        except PendingRollbackError:
            session.rollback()
        # session.rollback()
        return session

    def load_sql_to_rel_cache_db(self, ref):
        if self.incremental_manual_examples is not None:
            res = (self.sql_to_rel_cache_db.query(ChatGPTSQLToRELResponseIncremental).
                   filter_by(kaggle_ref_incremental=ref+str(self.incremental_manual_examples)).first())
        else:
            res = self.sql_to_rel_cache_db.query(ChatGPTSQLToRELResponse).filter_by(kaggle_ref=ref).first()
        if res is None:
            return None
        else:
            res = json.loads(res.openai_response)
            return res

    def write_sql_to_rel_cache_db(self, ref, res, session=None):
        res_str = json.dumps(res)
        if self.incremental_manual_examples is not None:
            new_response = ChatGPTSQLToRELResponseIncremental(
                kaggle_ref_incremental=ref+str(self.incremental_manual_examples),
                                                   openai_response=res_str)
        else:
            new_response = ChatGPTSQLToRELResponse(kaggle_ref=ref, openai_response=res_str)
        if session is None:
            session = self.sql_to_rel_cache_db
        session.add(new_response)
        try:
            session.commit()
        except PendingRollbackError:
            session.rollback()
            raise

    def rm_sql_to_rel_cache_db(self, ref):
        res = self.sql_to_rel_cache_db.query(ChatGPTSQLToRELResponse).filter_by(kaggle_ref=ref).first()
        self.sql_to_rel_cache_db.delete(res)
        self.sql_to_rel_cache_db.commit()

    @timeout(10)
    def make_sql_with_timeout(self, row):
        return self.make_sql(row)

    def select_rows_and_total(self, total = 200):
        catalog = self.get_catalog()
        minimum_executable_sqls=10
        rows = []
        while len(rows)< total:
            ds_iloc = random.randrange(0, len(catalog))
            row =  catalog.iloc[ds_iloc]

            flags = self.sql_executable_flag[ds_iloc]
            if flags is not None:
                bool_flags = [flag == "Valid" for flag in flags]
                if sum(bool_flags) >= minimum_executable_sqls:
                    rows.append((ds_iloc, row))
        return rows
    
    def manual_select_rows_and_total(self, indices):
        catalog = self.get_catalog()
        rows = []
        for ds_iloc in indices:
            row =  catalog.iloc[ds_iloc]
            rows.append((ds_iloc, row))
        return rows
    def make_sql(self, row, db_name=None):
        name = row['ref']
        csvs, sqlites = self.kaggle.find_files(name)
        sql = SQLMagic(db_name)
        try:
            if len(sqlites) > 0:
                # some datasets have both sqlite and csv files, prioritize sqlite
                sql.load_sqlite_dataset(sqlites)
            elif len(csvs) > 0:
                sql.load_csv_dataset(csvs)
            else:
                raise NoDataFound()
        except (sqlalchemy.exc.OperationalError, UnicodeDecodeError, pandas.errors.ParserError,
                pandas.errors.EmptyDataError, sqlalchemy.exc.ArgumentError) as e:
            raise NoDataFound()
        return sql

    def get_all_schemas_and_top_rows(self, step):
        if self.schema_path.exists() and step > 2:
            with self.schema_path.open('r') as f:
                st = f.read()
                schemas, raw_schemas = json.loads(st)
            if len(schemas) == len(self.get_catalog()):
                self.schemas = schemas
                self.raw_schemas = raw_schemas
                return
        catalog = self.get_catalog()
        schemas = []
        raw_schemas = []
        top_rows = []
        good_schema_cnt = 0
        # this loop needs to be parallelized
        pbar = tqdm(catalog.iloc, total=len(catalog))
        for row_idx, row in enumerate(pbar):
            try:
                if row_idx in (64839, 64840):
                    raise Exception
                sql_magic = self.make_sql_with_timeout(row)
                schema, raw_schema = sql_magic.get_all_tables_schema(string=False)
                top_row = sql_magic.get_all_tables_first_row_data()
                good_schema_cnt += 1
                pbar.set_description(desc=f"Good schema: {good_schema_cnt / (row_idx + 1):.1%}")
            except:
                schema = None
                top_row = None
                raw_schema = None
            schemas.append(schema)
            raw_schemas.append(raw_schema)
            top_rows.append(top_row)
        with self.schema_path.open('w') as f:
            f.write(json.dumps((schemas, raw_schemas)))
        with self.top_row_path.open('w') as f:
            f.write(json.dumps(top_rows))
        self.schemas = schemas
        self.raw_schemas = raw_schemas

    def chatgpt_synthesize_sql_queries(self, step):
        if self.synthesized_sql_path.exists() and step > 3:
            with self.synthesized_sql_path.open('r') as f:
                st = f.read()
                synthesized_sql = json.loads(st)
        else:
            catalog = self.get_catalog()
            schemas = self.schemas
            with self.top_row_path.open('r') as f:
                top_rows = json.loads(f.read())
            assert len(schemas) == len(top_rows) == len(catalog)
            synthesized_sql = self.chatgpt.synthesize_all_sql_code(catalog, schemas, top_rows)
            with self.synthesized_sql_path.open('w') as f:
                f.write(json.dumps(synthesized_sql))
        self.synthesized_sql = synthesized_sql

    def clear_executable_sql_flag(self):
        ExecutableFlag.__table__.drop(self.sql_to_rel_cache_db.bind)
        ExecutableFlag.__table__.create(self.sql_to_rel_cache_db.bind)

    def start_executing_sql(self, ref):
        start = ExecutableFlag(ref=ref, executed=False, executable_flag="")
        self.sql_to_rel_cache_db.add(start)
        self.sql_to_rel_cache_db.commit()

    def find_prev_execution(self, ref):
        try:
            exist = self.sql_to_rel_cache_db.query(ExecutableFlag).filter_by(ref=ref).first()
        except sqlalchemy.exc.OperationalError:
            return False
        if exist:
            if not exist.executed:
                return "OOM"
            else:
                return json.loads(exist.executable_flag)
        else:
            return False

    def insert_executable_flag(self, ref, executable_flag):
        exist = self.sql_to_rel_cache_db.query(ExecutableFlag).filter_by(ref=ref).first()
        if exist:
            exist.executed = True
            exist.executable_flag = json.dumps(executable_flag)
        else:
            ef = ExecutableFlag(ref=ref, executed=False, executable_flag=json.dumps(executable_flag))
            self.sql_to_rel_cache_db.add(ef)

        self.sql_to_rel_cache_db.commit()

    def write_rel_execution_results(self, ref, acc, infos, sql_times, rel_times, problems):
        try:
            if self.incremental_manual_examples is None:
                ef = RelExecutionRerun(ref=ref, acc=json.dumps(acc), infos=json.dumps(infos),
                                       sql_time=json.dumps(sql_times), rel_time=json.dumps(rel_times),
                                       problems=json.dumps(problems))
            else:
                ef = RelExecutionRerunIncremental(ref_incremental=ref+str(self.incremental_manual_examples), acc=json.dumps(acc), infos=json.dumps(infos),
                                       sql_time=json.dumps(sql_times), rel_time=json.dumps(rel_times),
                                       problems=json.dumps(problems))
            self.sql_to_rel_cache_db.add(ef)
            self.sql_to_rel_cache_db.commit()
        except PendingRollbackError:
            traceback.print_exc()
            print("Rollback!")
            self.sql_to_rel_cache_db.rollback()
        except sqlite3.IntegrityError:
            traceback.print_exc()
            print("Exception ignored")

    def find_rel_execution_results(self, ref):
        if self.incremental_manual_examples is None:
            exist = self.sql_to_rel_cache_db.query(RelExecutionRerun).filter_by(ref=ref).first()
        else:
            exist = (self.sql_to_rel_cache_db.query(RelExecutionRerunIncremental)
                     .filter_by(ref_incremental=ref+str(self.incremental_manual_examples)).first())
        if exist is None:
            return None
        acc, infos = exist.acc, exist.infos
        return json.loads(acc), json.loads(infos)

    def filter_sql_execution_with_ground_truth(self, step):
        """
        We keep the dataset with at least 5 executable SQL queries
        """
        if self.executable_flags_path.exists() and step > 4:
            with self.executable_flags_path.open('r') as f:
                st = f.read()
                executable_flags = json.loads(st)
        else:
            catalog = self.get_catalog()
            schemas = self.schemas
            synthesized_sql = self.synthesized_sql
            pbar = tqdm(catalog.iloc, total=len(catalog), desc="Filtering SQL execution with ground truth")
            executable_flags = []
            working_count = 0
            working_total = 0
            # self.clear_executable_sql_flag()
            for row_idx, row in enumerate(pbar):
                sqls = synthesized_sql[row_idx]
                schema = schemas[row_idx]
                if schema is not None and sqls is not None:
                    descriptions, sqls = zip(*sqls)
                    if sqls is not None:
                        ref = row['ref']
                        prev_res = self.find_prev_execution(ref)
                        if prev_res == "OOM":
                            working = None
                        elif prev_res is False:
                            self.start_executing_sql(ref)
                            sql_magic = self.make_sql(row)
                            try:
                                # schema only for debugging
                                results = sql_magic.execute_query_with_timeout(sqls, schema=schema)
                                # it will take too much space to store the execution results. only a boolena flag
                                # working = [r is not None and len(r[0]) != 0 for r in results]
                                working = []
                                for r in results:
                                    if r is not None and len(r[0]) != 0:
                                        working.append("Valid")
                                    elif r is None:
                                        working.append("Error")
                                    else:
                                        working.append("Empty")
                                self.insert_executable_flag(ref, working)
                            except:
                                working = None
                        else:
                            working = prev_res
                    else:
                        working = None
                else:
                    working = None
                if working is not None:
                    wc = [w == "Valid" for w in working]
                    working_count += sum(wc) if isinstance(wc, list) else 0
                working_total += self.chatgpt.sql_per_ds
                pbar.set_description(
                    f"Filtering SQL execution with ground truth, {working_count / working_total:.2f} acc")
                executable_flags.append(working)
            with self.executable_flags_path.open('w') as f:
                f.write(json.dumps(executable_flags))
        self.sql_executable_flag = executable_flags

    def get_catalog(self):
        catalog = self.kaggle.all_catalog_datasets()
        if self.dev:
            catalog = catalog.iloc[:self.dev_slice]
        return catalog

    def find_sqls(self):
        catalog = self.kaggle.all_catalog_datasets()
        # pbar = tqdm(catalog.iloc, total=len(catalog))

        while True:
            ds_idx = random.randrange(0, len(catalog) - 1)
            row = catalog.iloc[ds_idx]
            sqls = self.synthesized_sql[ds_idx]
            flags = self.sql_executable_flag[ds_idx]
            if sqls is not None and flags is not None:
                sql_idx = random.randrange(0, len(sqls) - 1)
                if not flags[sql_idx]:
                    print(sqls[sql_idx])
                    schemas = self.schemas[ds_idx]
                    sql_magic = self.make_sql(row)
                    q = sqls[sql_idx][1]
                    q = q.replace("code:", "")
                    q = q.replace("Code:", "")
                    try:
                        results = sql_magic.execute_query(q)
                        print("HEREIOJOI")
                    except:
                        pass

    def develop_sql_rel_examples(self, step):
        # def wanted_sql(sql):
        #     return True
        #     return "COUNT" in sql[1] and len(sql[1]) > 100

        if step == 5 and self.dev:
            catalog = self.kaggle.all_catalog_datasets()
            pbar = tqdm(catalog.iloc, total=len(catalog))

            for ds_idx, row in enumerate(pbar):
                # if row['ref'] != "konradb/us-military-interventions":
                #     continue
                # if ds_idx <= 7:  # 11:
                #     continue

                # if len(csv_files) < 2:
                #     continue

                # if sqls is None:
                #     continue

                # if not any([wanted_sql(q) for q in sqls]):
                #     continue

                sqls = self.synthesized_sql[ds_idx]
                flags = self.sql_executable_flag[ds_idx]

                # table_loading_code, number_unique_name_to_readable_name, readable_name_to_number_unique_name = \
                #     self.upload_dataset_to_rai(self.relx, row, ds_idx)

                # self.relx.execute_rel_query_against_all_dataset_tables(query, number_unique_name_to_readable_name)
                for sql_idx, q in enumerate(sqls):
                    print(sql_idx)
                    print(q)
                    break
                schemas = self.schemas[ds_idx]
                sql_magic = self.make_sql(row)
                q = sqls[sql_idx][1]
                q = q.replace("code:", "")
                q = q.replace("Code:", "")
                results = sql_magic.execute_query(q)
                print("HEREIOJOI")

                # sql_magic = self.make_sql(row)
                # q = 'SELECT p.imagelink \n' \
                #     'FROM Photos p \n' \
                #     'JOIN Likes l ON p.id = l.photo\n' \
                #     'JOIN Users u ON l.user = u.id\n' \
                #     'LEFT JOIN Follows f ON u.id = f.follower AND f.followee = p.userID\n' \
                #     # 'WHERE f.follower IS NULL;'
                # results = sql_magic.execute_query(q)
                # print("HEREIOJOI")

    # results = sql_magic.execute_query(q)
    #
    # L-management:
    # our goal is to finish the pipeline ASAP.
    # we start from this current entry, push the data to the database, and we run manually written queries
    # get 10 that runs, and they will be prompts

    def upload_dataset_to_rai(self, relx, catalog_row, ds_idx):
        name = catalog_row['ref']
        csv_files = self.kaggle.find_files(name)[0]

        number_unique_name_to_readable_name = {}
        readable_name_to_number_unique_name = {}
        for csv_idx, csv_file in enumerate(csv_files):
            tbl_name = csv_file.stem
            tbl_name = self.sql.table_name_sanitize(tbl_name)
            number_unique_name = f"ds_{ds_idx}_tbl_{csv_idx}"

            df = pandas.read_csv(csv_file)
            if 'Unnamed: 0' in df.columns:
                df = df.drop('Unnamed: 0', axis=1)
            df = df.rename(columns=self.sql.rename_columns(df.columns))
            csv_string = io.StringIO()
            df.to_csv(csv_string)
            csv_string = csv_string.getvalue()
            schema = self.get_schema_from_sql(ds_idx, tbl_name)
            relx.upload_csv(csv_string, number_unique_name, schema)
            number_unique_name_to_readable_name[number_unique_name] = tbl_name
            readable_name_to_number_unique_name[tbl_name] = number_unique_name

        # execute rel query
        table_loading_code = ""
        for number_unique_name, readable_name in number_unique_name_to_readable_name.items():
            table_loading_code += f"def {readable_name} = {number_unique_name}\n"

        return table_loading_code, number_unique_name_to_readable_name, readable_name_to_number_unique_name

    def get_schema_from_pandas_df(self, df):
        # https://docs.relational.ai/rel/how-to/csv-import#schema
        schema = {}
        for k, v in df.dtypes.to_dict().items():
            if v.name == "object":
                schema[k] = "string"
            elif v.name == "float64":
                schema[k] = "float"
            elif v.name == "int64":
                schema[k] = "int"
            elif v.name == "bool":
                schema[k] = "bool"
            else:
                raise NotImplementedError
        return schema

    def get_schema_from_sql(self, iloc, tbl_name):
        # not_null_row = self.get_catalog().index.get_loc(ds_idx)
        raw_schemas = self.raw_schemas[iloc]
        raw_schema = raw_schemas[tbl_name]
        schema = {}
        for k, v in raw_schema:
            if v == "TEXT":
                schema[k] = "string"
            elif v == "FLOAT":
                schema[k] = "float"
            elif v == "INT":
                schema[k] = "int"
            elif v == "BIGINT":
                schema[k] = "int"
            elif v == "BOOLEAN":
                schema[k] = "bool"
            else:
                raise NotImplementedError
        return schema

    def get_expected_has_rel_datasets(self):
        pipeline = self
        result = pipeline.sql_executable_flag

        none_percentage = len([r for r in result if r is None]) / len(result)
        true_percentage_not_none = 0
        true_percentage_total = 0
        not_none_idx = []
        for ridx, r in enumerate(result):
            if r is not None:
                true_percentage_not_none += sum(r)
                true_percentage_total += len(r)
                not_none_idx.append(ridx)
        true_percentage_not_none = true_percentage_not_none / true_percentage_total
        rel_cnt = 0
        rel_ds = 0
        for ridx, r in enumerate(result):
            if r is not None:
                if sum(r) >= 10:
                    rel_ds += 1
                    rel_cnt += 10
        return rel_ds

    def dump_sql_to_rel_dataset(self, step):
        if step == 6.1:
            caches = []
            catalog = self.get_catalog()
            for ds_iloc, one_row in tqdm(enumerate(catalog.iloc)):
                cache = self.load_sql_to_rel_cache_db(one_row['ref'])
                caches.append(cache)
            all_raw_answers = []
            all_synths = []
            for c in caches:
                if c is None:
                    pass
                else:
                    all_synths.append(c[2])
                    all_raw_answers.append(c[1])
            # n = 25000
            # print(catalog.iloc[all_synths[n][0]['idx'][0]])
            # print(all_synths[n])

            with self.sql_to_rel_translated_path.open('w') as f:
                f.write(json.dumps((all_synths, all_raw_answers)))
    def chatgpt_translate_sql_to_rel(self, step, dump=False, selected_indices_and_rows = None):
        if self.sql_to_rel_translated_path.exists() and step > 6:
            with self.sql_to_rel_translated_path.open('r') as f:
                st = f.read()
                all_synths, all_raw_answers = json.loads(st)
        else:
            chatgpt = KagglePrompting()
            catalog = self.get_catalog()
            if selected_indices_and_rows is None:
                rows = catalog.iloc
                total = self.get_expected_has_rel_datasets()
            else:
                rows = selected_indices_and_rows
                total = len(selected_indices_and_rows)
            pbar = tqdm(rows, total=total, desc="ChatGPT translating SQL to REL")
            minimum_executable_sqls = 10
            total_workers = 3
            future_answers = []
            ready_queue = Queue()
            for i in range(total_workers):
                ready_queue.put(self.connect_sql_to_rel_cache_db())
            exclude_ground_truth_refs = set([q['ref'] for q in
                                             KagglePrompting.manual_examples(self.incremental_manual_examples)])
            cnt = 0
            with ThreadPoolExecutor(total_workers) as executor:
                iter = enumerate(rows) if selected_indices_and_rows is None else rows
                for ds_iloc, row in iter:
                    sqls = self.synthesized_sql[ds_iloc]
                    flags = self.sql_executable_flag[ds_iloc]
                    if row['ref'] in exclude_ground_truth_refs:
                        continue
                    if flags is None:
                        continue
                    if type(flags[0]) is not str:
                        raise RuntimeError("Flags types is changed.")
                    else:
                        bool_flags = [f=='Valid' for f in flags]
                    if sum(bool_flags) >= minimum_executable_sqls:
                        # a batch always belongs to the same dataset
                        batch = []
                        cnt += 1
                        # if cnt < 28660:
                        #     continue
                        for sql_idx, (desc_sql, sql_executable) in enumerate(zip(sqls, bool_flags)):
                            if sql_executable:
                                batch.append({
                                    "idx": (ds_iloc, sql_idx),
                                    "query": desc_sql[1]
                                })
                                if len(batch) == minimum_executable_sqls:
                                    break
                        db_key = row['ref'] if self.incremental_manual_examples is None else row['ref'] + str(
                            self.incremental_manual_examples)

                        cache = self.load_sql_to_rel_cache_db(db_key)
                        kwargs = {
                            "pipeline": self,
                            "chatgpt": chatgpt,
                            "cache": cache,
                            "row": row,
                            "batch": batch,
                            "ready_queue": ready_queue,
                            "pbar": pbar
                        }
                        # worker_translate_sql_to_rel(**kwargs)
                        future = executor.submit(worker_translate_sql_to_rel, **kwargs)
                        future_answers.append(future)

                        # if cache is None:
                        #     raw_answer, synths = chatgpt.proto_synthesis_one_batch(batch, verbose_level=0)
                        #     self.write_sql_to_rel_cache_db(row['ref'], (batch, raw_answer, synths))
                        # else:
                        #     last_batch, raw_answer, synths = cache
                        #     last_batch_idx = tuple(tuple(l['idx']) for l in last_batch)
                        #     batch_id = tuple(tuple(l['idx']) for l in batch)
                        #     if last_batch_idx != batch_id:
                        #         raw_answer, synths = chatgpt.proto_synthesis_one_batch(batch, verbose_level=0)
                        #         self.rm_sql_to_rel_cache_db(row['ref'])
                        #         self.write_sql_to_rel_cache_db(row['ref'], (batch, raw_answer, synths))
                        # backup_file = project_root() / f"cache/sql_to_rel/{row['ref']}.json"
                        # backup_file.parent.mkdir(parents=True, exist_ok=True)
                        # with open(backup_file, 'w') as f:
                        #     f.write(json.dumps((raw_answer, synths)))
            concurrent.futures.wait(future_answers)
            answers = [fut.result() for fut in future_answers]
            all_raw_answers = [e[0] for e in answers]
            all_synths = [e[1] for e in answers]
            with self.sql_to_rel_translated_path.open('w') as f:
                f.write(json.dumps((all_synths, all_raw_answers)))
                
        self.sql_to_rel_translated = all_synths
        self.sql_to_rel_translated_raw_answers = all_raw_answers

    @property
    def rel_execution_accuracy_path(self):
        st = "" if self.incremental_manual_examples is None else f"_{self.incremental_manual_examples}"
        return kaggle_path / f"rel_execution_accuracy{self.dev_st}_rerun{st}.json"

    @property
    def sql_to_rel_translated_path(self):
        if self.incremental_manual_examples is None:
            return kaggle_path / f"sql_to_rel_translated{self.dev_st}.json"
        else:
            return kaggle_path / f"sql_to_rel_translated{self.dev_st}_{self.incremental_manual_examples}.json"

    def split_relation_id(self, relation_id):
        # e.g. '/:output/FilePos/FilePos/String/String' or '/:output/:AverageMontlyHours/FilePos/Int64'
        lst = relation_id.split("/")
        cols_flag = False
        tbl = None
        cols = []
        last_col_idx = 0
        for idx, l in enumerate(lst):
            if ":" in l:
                last_col_idx = idx
                tbl = l.strip(":")

        cols = lst[last_col_idx + 1:]
        return tbl, cols

    def compare_sql_rel_results(self, responses, sql_results, rels, sql_code, rel_code, row=None, sql_magic=None):
        corrects = []
        infos = []
        problems = []
        for i in range(len(responses)):
            rsp = responses[i]
            rel = rel_code[i]
            sql = sql_code[i]
            sql_result = sql_results[i]
            if rsp is not None and len(rsp.problems) == 0:
                try:
                    if len(rsp.results) == 1:
                        sql_joined = pandas.DataFrame(sql_result[0])
                        rel_results = []
                        res = rsp.results[0]
                        tbl = res["table"].to_pandas()
                        cols = res['relationId']
                        columns = cols.split("/")[2:]
                        indices = [i for i, x in enumerate(columns) if x == "FilePos"]
                        for file_pos_index in indices:
                            tbl = tbl.drop(f"v{file_pos_index + 1}", axis=1)
                        rel_results.append(tbl)
                    else:
                        sql_joined = pandas.DataFrame(sql_result[0])
                        rel_results = []
                        with_missing = {}
                        sql_column_sort_flag = False
                        for j, res in enumerate(rsp.results):
                            rid = res['relationId']
                            tbl_name, columns = self.split_relation_id(rid)

                            tbl = res["table"].to_pandas()
                            if rid.count(":") == 1:
                                indices = [i for i, x in enumerate(columns) if x == "FilePos"]
                                for file_pos_index in indices:
                                    tbl = tbl.drop(f"v{file_pos_index + 1}", axis=1)
                            else:
                                # e.g. '/:output/:WorkAccident/FilePos/Int64'
                                tbl = tbl.set_index('v1')
                                tbl = tbl.rename({"v2": tbl_name}, axis=1)
                                sql_column_sort_flag = True

                            if tbl_name != "COL1":
                                if tbl_name in with_missing:
                                    with_missing[tbl_name].append(tbl)
                                else:
                                    with_missing[tbl_name] = [tbl]
                        if sql_column_sort_flag:
                            sql_joined = sql_joined.sort_index(axis=1)

                        # merge missing
                        for k, v in with_missing.items():
                            if len(v) > 1:
                                for t in v:
                                    for c in t.columns:
                                        if isinstance(t[c].iloc[0], dict):
                                            t[c] = None
                                tbl = pandas.concat(v)
                            else:
                                tbl = v[0]
                            rel_results.append(tbl)
                    # if True:
                    rel_joined = pandas.concat(rel_results, axis=1)
                    # join_size = 1
                    # for r in rel_results:
                    #     join_size *= len(r)
                    # # if join_size > 1e9:
                    # #     raise
                    # rel_joined = rel_results[0]
                    # for i in rel_results[1:]:
                    #     rel_joined = rel_joined.join(i, how='outer', sort=True)

                    # rel_joined = rel_results[0].join(rel_results[1:], how='outer', sort=True)
                    rel_joined = rel_joined.reset_index(drop=True)
                    sql_joined = sql_joined.reset_index(drop=True)

                    if False:
                        diff_cols = set(rel_joined.columns).symmetric_difference(set(sql_joined.columns))

                    rename = {
                        rel_joined.columns[i]:
                            f"{rel_joined.columns[i]}_{sql_joined.columns[i]}" for i in range(len(sql_joined.columns))
                    }
                    rel_joined = rel_joined.rename(rename, axis=1)
                    # rename = {
                    #     sql_joined.columns[i]:
                    #         f"{rel_joined.columns[i]}" for i in range(len(sql_joined.columns))
                    # }
                    # sql_joined = sql_joined.rename(rename, axis=1)
                    sql_joined.columns = list(rename.values())
                    rel_joined = rel_joined.reset_index(drop=True)
                    sql_joined = sql_joined.reset_index(drop=True)
                    try:
                        diff = rel_joined.compare(sql_joined)
                    except ValueError:
                        sql_joined = sql_joined.dropna(axis=1, how='all')
                        diff = rel_joined.compare(sql_joined)

                    correct = diff.empty
                    if not correct:
                        rel_joined = rel_joined.applymap(lambda x: None if x == {} else x)
                        diff = rel_joined.compare(sql_joined)
                        correct = diff.empty
                    if not correct:
                        rel_joined = rel_joined.sort_values(by=rel_joined.columns.to_list())
                        sql_joined = sql_joined.sort_values(by=sql_joined.columns.to_list())
                        rel_joined = rel_joined.reset_index(drop=True)
                        sql_joined = sql_joined.reset_index(drop=True)
                        diff = rel_joined.compare(sql_joined)
                        correct = diff.empty
                    if not correct:
                        all_correct = True
                        for col in sql_joined.columns:
                            if sql_joined[col].dtypes in ["int", "float"]:
                                all_correct = all_correct and np.isclose(sql_joined[col], rel_joined[col]).all().item()
                            else:
                                all_correct = all_correct and (sql_joined[col] == rel_joined[col]).all().item()
                        correct = all_correct
                except:
                    correct = False
                if correct:
                    infos.append("correct")
                    problems.append(None)
                else:
                    if len(rsp.results) == 0:
                        infos.append("error")
                        problems.append("empty response")
                    else:
                        infos.append('comparison')
                        problems.append(None)
                corrects.append(correct)
            else:
                corrects.append(False)
                infos.append('error')
                if rsp is not None:
                    problems.append(rsp.problems)
                else:
                    problems.append(None)
        return corrects, infos, problems

    def execute_rel_accuracy(self, step):
        if self.rel_execution_accuracy_path.exists() and step > 7:
            with self.rel_execution_accuracy_path.open('r') as f:
                st = f.read()
                execution_accuracy = json.loads(st)
        else:
            http_error_cnt = 10
            total_workers = 10
            catalog = self.get_catalog()
            ready_synth_pipeline_queue = Queue()
            for i in range(total_workers):
                synth_pipeline = SynthesisPipeline(dev=self.dev, relx_identifier=f"{i}",
                                                   incremental_manual_examples = self.incremental_manual_examples,
                                                   make_engine=True)
                synth_pipeline.get_all_schemas_and_top_rows(step=3)
                ready_synth_pipeline_queue.put(synth_pipeline)
            future_accuracies = []
            correct = 0
            total = 0
            with ThreadPoolExecutor(total_workers) as executor:
                # tbar = tqdm(total=len(self.sql_to_rel_translated))
                tbar = None
                batch_future = []
                for i, same_ds_synth in enumerate(self.sql_to_rel_translated):
                    try:
                        iloc = same_ds_synth[0]['idx'][0]
                    except (IndexError, TypeError) as e:
                        continue

                    row = catalog.iloc[iloc]
                    res = self.find_rel_execution_results(row['ref'])
                    sqls = self.synthesized_sql[iloc]
                    kwargs = {
                        "row": row,
                        "iloc": iloc,
                        "sqls": sqls,
                        "same_ds_synth": same_ds_synth,
                        "ready_synth_pipeline_queue": ready_synth_pipeline_queue,
                        "res": res,
                    }
                    future = executor.submit(upload_and_run_thread_function, **kwargs)
                    future_accuracies.append(future)
                    batch_future.append(future)
                    if len(batch_future) > total_workers:
                        old_len = len(batch_future)
                        concurrent.futures.wait(batch_future, return_when='FIRST_COMPLETED')
                        done_future = []
                        going_future = []
                        for fut in batch_future:
                            if fut.done():
                                done_future.append(fut)
                            else:
                                going_future.append(fut)
                        batch_errors = 0
                        for future in done_future:
                            if hasattr(future, '_exception'):
                                if future._exception is not None:
                                    batch_errors += 1
                                if isinstance(future._exception, RuntimeError):
                                    raise RuntimeError
                                if isinstance(future._exception, HTTPError):
                                    http_error_cnt -= 1
                                    if http_error_cnt == 0:
                                        raise HTTPError
                            ref, acc, info = fut.result()
                            if acc is None:
                                pass
                            else:
                                total += len(acc)
                                correct += sum(acc)
                        if batch_errors > 3 and batch_errors / len(done_future) > 0.5:
                            raise RuntimeError("High error percentage. Terminating execution.")
                        batch_future = going_future
                        new_len = len(batch_future)
                        if tbar is not None:
                            tbar.update(old_len - new_len)
                            tbar.set_description(desc=f"Executing REL for ground truth accuracy {correct / total}")
                        else:
                            print(f"{total} out of {len(self.sql_to_rel_translated) * 10} with acc {correct / total}")

                    # print("1")
            concurrent.futures.wait(future_accuracies)
            execution_accuracy = [fut.result() for fut in future_accuracies]
            refs = [e[0] for e in execution_accuracy]
            all_accs = [e[1] for e in execution_accuracy]
            all_infos = [e[2] for e in execution_accuracy]
            acc = sum([a for a in all_accs if a is not None], [])
            print(f"Average accuracy {sum(acc) / len(acc)}")
            with self.rel_execution_accuracy_path.open('w') as f:
                st = json.dumps([refs, all_accs, all_infos])
                f.write(st)
        self.rel_execution_accuracy = execution_accuracy

    def dev_verify_ground_truth_acc(self, step, examples=None):
        if step == 7.1:
            self.dev_slice = -1
            catalog = self.get_catalog()
            total_workers = 1
            ready_synth_pipeline_queue = Queue()
            for i in range(total_workers):
                synth_pipeline = SynthesisPipeline(dev=self.dev, relx_identifier=f"{i}")
                synth_pipeline.get_all_schemas_and_top_rows(step=3)
                ready_synth_pipeline_queue.put(synth_pipeline)
            future_accuracies = []
            with ThreadPoolExecutor(total_workers) as executor:
                # tbar = tqdm(total=len(self.sql_to_rel_translated))
                tbar = None
                if examples is None:
                    examples = KagglePrompting.manual_examples()
                for i, ex in enumerate(examples):
                    ref, sql, rel = ex['ref'], ex['sql'], ex['rel']
                    # row = catalog.loc[catalog['ref'] == ref]
                    iloc = np.where((catalog['ref'] == ref))[0][0]
                    # row, idx = row.iloc[0], row.index[0]
                    row = catalog.iloc[iloc]
                    sql_magic = self.make_sql(row)
                    rels = [rel]
                    sql_results = []
                    submitted_sqls = []
                    sql_result = sql_magic.execute_query(sql)
                    sql_results.append(sql_result)
                    submitted_sqls.append(("", sql))
                    kwargs = {
                        "row": row,
                        "iloc": iloc,
                        "sqls": submitted_sqls,
                        "same_ds_synth": rels,
                        "ready_synth_pipeline_queue": ready_synth_pipeline_queue,
                        "res": None,
                        "write_to_cache": False,
                    }
                    future = executor.submit(upload_and_run_thread_function, **kwargs)
                    future_accuracies.append(future)
                    sql_magic.terminate()
            concurrent.futures.wait(future_accuracies)
            execution_accuracy = [fut.result() for fut in future_accuracies]
            return execution_accuracy

    def inspect_results(self, step):
        if step == 7.2:
            self.relx.make_engine()
            with self.rel_execution_accuracy_path.open('r') as f:
                acc, infos = json.loads(f.read())
            for i, same_ds_synth in enumerate(self.sql_to_rel_translated):
                try:
                    iloc = same_ds_synth[0]['idx'][0]
                except IndexError:
                    continue
                catalog = self.get_catalog()
                row = catalog.iloc[iloc]
                sql_magic = self.make_sql(row)
                sqls = self.synthesized_sql[iloc]
                rels = []
                sql_results = []
                submitted_sqls = []
                for sidx, synth in enumerate(same_ds_synth):
                    _, sql_idx = synth['idx']
                    desc, sql = sqls[sql_idx]
                    sql_result = sql_magic.execute_query(sql)
                    sql_results.append(sql_result)
                    rels.append(synth['query'])
                    submitted_sqls.append((desc, sql))
                    kwargs = {
                        "row": row,
                        "iloc": iloc,
                        "sql_magic": None,
                        "rels": rels,
                        "sql_results": sql_results,
                        "submitted_sqls": submitted_sqls,
                        "acc": acc[i][sidx],
                        "info": infos[i][sidx]
                    }
                    table_loading_code, number_unique_name_to_readable_name, readable_name_to_number_unique_name, schemas = \
                        upload_to_rai(row, iloc, self, self.relx)
                    if kwargs['info'] == "syntax":
                        print(f"==============={i}-{sidx}===================")
                        print(desc)
                        print()
                        print(sql)
                        print()
                        print(rels[sidx])
            self.relx.shutdown_engine()

    def export_all_datasets(self, step):
        """

        :return:
        """
        # must run manually to prevent overwrite.
        if step != 8:
            return

        export_path = kaggle_path / "export4"
        if not export_path.exists():
            export_path.mkdir()

        # 1 
        # the catalog of kaggle datasets
        catalog = self.get_catalog()

        # the schemas extracted and top rows
        schemas = self.schemas
        raw_schemas = self.raw_schemas
        with self.top_row_path.open('r') as f:
            top_rows = json.loads(f.read())
        assert len(schemas) == len(top_rows) == len(catalog)

        # the sql code synthesized by chatgpt
        sql = self.synthesized_sql

        # whether the sql code is executable with ground truth
        sql_executable = self.sql_executable_flag

        catalog["schemas"] = schemas
        catalog['raw_schemas'] = raw_schemas
        catalog['top_rows'] = top_rows
        catalog["sql_translated"] = sql
        catalog["sql_executable"] = sql_executable

        # catalog_path = export_path / "catalog.jsonl"
        # with open(catalog_path, 'w') as f:
        #     st = catalog.to_json(orient='records', lines=True)
        #     f.write(st)
        catalog_path = export_path / "catalog.json"
        with open(catalog_path, 'w') as f:
            st = catalog.to_json(orient='records', )
            f.write(st)
        # catalog_path = export_path / "catalog.csv"
        # with open(catalog_path, 'w') as f:
        #     st = catalog.to_csv()
        #     f.write(st)
        # 2 
        catalog = self.get_catalog()

        # chatgpt translations from sql to rel
        sql_to_rel_translated = self.sql_to_rel_translated
        sql_to_rel_translated_raw_answers = self.sql_to_rel_translated_raw_answers
        assert len(sql_to_rel_translated) == len(sql_to_rel_translated_raw_answers)

        # whether the rel code is executable
        all_rel_executable = self.rel_execution_accuracy
        all_rel_executable = list(zip(*all_rel_executable))
        cnt = 0
        df = {
            "ref": [],
            "sql_code": [],
            "rel_code": [],
            "sql_indices": [],
            "raw_chatgpt_output": [],
            "rel_executable": [],
            "rel_executable_info": [],
        }
        for i, same_ds_synth in enumerate(sql_to_rel_translated):
            if len(same_ds_synth) == 0:
                continue
            iloc = same_ds_synth[0]['idx'][0]
            row = catalog.iloc[iloc]
            ref = row['ref']
            rel_ex = all_rel_executable[cnt]
            assert rel_ex[0] == ref

            sqls = self.synthesized_sql[iloc]
            sql_indices = [s['idx'][1] for s in same_ds_synth]
            sql_code = [sqls[s] for s in sql_indices]
            rel_code = [s['query'] for s in same_ds_synth]

            rel_executable = rel_ex[1]
            rel_executable_info = rel_ex[2]

            df["ref"].append(ref)
            df['sql_code'].append(sql_code)
            df["rel_code"].append(rel_code)
            df['sql_indices'].append(sql_indices)
            df['raw_chatgpt_output'].append(sql_to_rel_translated_raw_answers[i])
            df['rel_executable'].append(rel_executable)
            df['rel_executable_info'].append(rel_executable_info)
            cnt += 1
        df = pandas.DataFrame(df)
        # sql_rel_path = export_path / "sql_rel.jsonl"
        # with open(sql_rel_path, 'w') as f:
        #     st = df.to_json(orient='records', lines=True)
        #     f.write(st)
        sql_rel_path = export_path / "sql_rel.json"
        with open(sql_rel_path, 'w') as f:
            st = df.to_json(orient='records', )
            f.write(st)
        # sql_rel_path = export_path / "sql_rel.csv"
        # with open(sql_rel_path, 'w') as f:
        #     st = df.to_csv()
        #     f.write(st)

        # 3
        # the manual sql to rel examples
        examples = KagglePrompting.manual_examples()
        examples = pandas.DataFrame(examples)
        # manual_examples_path = export_path / "manual_prompt.jsonl"
        # with open(manual_examples_path, 'w') as f:
        #     st = examples.to_json(orient='records', lines=True)
        #     f.write(st)
        manual_examples_path = export_path / "manual_prompt.json"
        with open(manual_examples_path, 'w') as f:
            st = examples.to_json(orient='records', )
            f.write(st)
        # manual_examples_path = export_path / "manual_prompt.csv"
        # with open(manual_examples_path, 'w') as f:
        #     st = examples.to_csv()
        #     f.write(st)

    def run_from_step(self, step=0):
        # step  0
        self.kaggle.all_catalog_datasets()
        # 1
        self.kaggle.download_datasets(step)
        # 2
        self.get_all_schemas_and_top_rows(step)
        # 3
        self.chatgpt_synthesize_sql_queries(step)
        # 4
        self.filter_sql_execution_with_ground_truth(step)
        # 5, skipped unless run directly in dev mode
        self.develop_sql_rel_examples(step)
        # 6
        self.chatgpt_translate_sql_to_rel(step)
        self.dump_sql_to_rel_dataset(step)
        # 7
        self.execute_rel_accuracy(step)
        # 7.1
        self.dev_verify_ground_truth_acc(step)
        # 7.2
        self.inspect_results(step)
        # 8
        self.export_all_datasets(step)


def check_catalog():
    pipeline = SynthesisPipeline()
    cat = pipeline.get_catalog()  # 22000 dataset


def get_more_dataset():
    kaggle = KaggleDownloader("2")
    kaggle.catalog_datasets()
    kaggle.download_datasets(1)


def runnow():
    step = 4
    pipeline = SynthesisPipeline()
    pipeline.kaggle.all_catalog_datasets()
    # 1
    pipeline.kaggle.download_datasets(step)
    # 2
    pipeline.get_all_schemas_and_top_rows(step)
    # 3
    pipeline.chatgpt_synthesize_sql_queries(step)
    # 4
    pipeline.filter_sql_execution_with_ground_truth(step)

def find_sqls():
    pipeline = SynthesisPipeline()
    pipeline.kaggle.all_catalog_datasets()
    step = 5
    # 1
    pipeline.kaggle.download_datasets(step)
    # 2
    pipeline.get_all_schemas_and_top_rows(step)
    # 3
    pipeline.chatgpt_synthesize_sql_queries(step)
    # 4
    pipeline.filter_sql_execution_with_ground_truth(step)
    pipeline.find_sqls()


def run_from_step_command_line():
    import sys
    start = sys.argv[1]
    pipeline = SynthesisPipeline()
    pipeline.run_from_step(int(start))

def export_annotated_tables_dataset():
    pipeline = SynthesisPipeline()
    pipeline.run_from_step(8)

if __name__ == '__main__':
    export_annotated_tables_dataset()
