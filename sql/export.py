import json
import random
import re

import pandas
import torch
from tqdm import tqdm

from llm.chatgpt import ChatGPT


class KaggleSqlRelSource:
    def __init__(self):
        self.catalog, self.sql_rel, self.manual_examples = self.load_exported_kaggle_sql_rel_source()
        sql_kws = ["group", "join", "where", 'from', 'like', 'in', 'avg', 'min', 'max', 'count', 'sum',
                   'or', 'and', 'not', 'order', 'having',
                   'intersect', 'union', 'limit', 'except', ]
        keywords = {
            kw: kw for kw in sql_kws
        }
        keywords['group'] = "group by"
        keywords['order'] = "order by"
        self.sql_keywords = keywords

    @staticmethod
    def load_exported_kaggle_sql_rel_source():
        from sql.synth import kaggle_path
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

        # all_rel_code = []
        # all_rel_code_executable = []
        # for idx, sql_rel_row in self.sql_rel.iterrows():
        #     if sql_rel_row['rel_code'] is not None and sql_rel_row['rel_executable'] is not None:
        #         all_rel_code.extend(sql_rel_row['rel_code'])
        #         all_rel_code_executable.extend(sql_rel_row['rel_executable'])
        #
        # indices = range(len(all_rel_code))
        # train_indices, valid_indices, test_indices = torch.utils.data.random_split(
        #     indices, [train_len, valid_len, test_len],
        #     generator=gen
        # )
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
            # code = [all_rel_code[i] for i in indices]
            # executable = [all_rel_code_executable[i] for i in indices]
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
        sql = sql.lower()
        sql = re.split(r'\W', sql)

        for kw, cl in self.sql_keywords.items():
            if kw in sql:
                cls.append(cl)
        return cls



if __name__ == '__main__':
    source = KaggleSqlRelSource()
    source.classified_accuracy()
