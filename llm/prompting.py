import asyncio
import json
import random
import re
import time

import openai
import ray
from tqdm import tqdm

from llm.chatgpt import ChatGPT
from rel.executor import RELX, table_id_to_rai_relation_name
from toolkit.util import project_root
import pandas


# some interactive development tool for myself


def upload_csv(csv_string, table_id, schema, relx=None):
    if relx is None:
        relx = RELX()
    relation_name = table_id_to_rai_relation_name(table_id, 'csv')
    relx.upload_csv(csv_string, relation_name, schema)
    return relation_name


def upload_json(json_string, table_id, relx=None):
    if relx is None:
        relx = RELX()
    relation_name = table_id_to_rai_relation_name(table_id, 'json')
    relx.upload_json(json_string, relation_name)
    return relation_name


def get_json_csv_string(dict_table):
    json_string = json.dumps(dict_table)
    df = pandas.DataFrame(dict_table)
    csv_string = df.to_csv()
    return json_string, csv_string




class WikiSQLPrompting:
    def __init__(self, engine_postfix=""):
        self.all_queries, self.all_schemas, self.all_tables = {}, {}, {}
        for split in "train", "dev", "test":
            self.all_queries[split], self.all_schemas[split], self.all_tables[split] = load_wikisql(split)
        self.relx = None
        self.split = None
        self.engine_postfix = engine_postfix
        self.cache_path = project_root() / "cache"
        self.load = load_wikisql

    def make_relx(self):
        self.relx = RELX(postfix=self.engine_postfix)
        self.relx.make_engine()
        self.relx.make_database()

    def set_split(self, split):
        self.split = split

    @property
    def queries(self):
        return self.all_queries[self.split]

    @property
    def schemas(self):
        return self.all_schemas[self.split]

    @property
    def tables(self):
        return self.all_tables[self.split]

    @staticmethod
    def manual_examples():
        queries = [
            {"idx": 41905,
             "query": 'def output[b] = tbl:col0(pos, a) and tbl:col2(pos, b) from a,pos where a = "scd037405362"'},
            {"idx": 7296,
             "query": 'def output[col1] = tbl:col1(pos, col1) and tbl:col6(pos, col6) and tbl:col3(pos, col3) from pos, col6, col3 where col6 = "allison b400r" and col3 = "brt"'},
            {"idx": 1639,
             "query": 'def x[col1] = tbl:col1(pos, col1) and tbl:col5(pos, col5) from pos, col5 where col5 = "2t6705"\n'
                      'def output = count[x]'},
            {"idx": 48598,
             "query": 'def x[a1] = tbl:col1(pos, a1) and tbl:col0(pos, a0) from pos, a0 where a0>1956.0 \n'
                      'def output = x'},
            {"idx": 18024,
             'query': 'def output[a2] = tbl:col2(pos, a2) and tbl:col0(pos, a0) from pos, a0 where a0 = "early september 1999"'
             },
            {"idx": 16049,
             'query': 'def output[x1] = tbl:col1(pos, x1) and tbl:col4(pos, c4) from pos, c4 where c4 = "yale university"'
             },
            {"idx": 14628,
             'query': 'def x[c5] = tbl:col5(pos, c5) and tbl:col1(pos, c1) from pos, c1 where c1 = "archduke karl"\n'
                      'def output = count[x]'
             },
            {"idx": 9144,
             'query': 'def x[c4] = tbl:col4(pos, c4) and tbl:col0(pos, c0) from pos, c0 where c0 = "baseball"\n'
                      'def output = min[x]'
             },
            {"idx": 48265,
             'query': 'def x[c2] = tbl:col2(pos, c2) and tbl:col0(pos, c0) from pos, c0 where c0 = "third round"\n'
                      'def output = max[x]'
             },
            {"idx": 6717,
             'query': 'def x[c5] = tbl:col5(pos, c5) and tbl:col4(pos, col4) from pos, col4 where col4 = "60.000"\n'
                      'def output = count[x]'
             }
        ]

        return queries

    def make_prototype_prompt(self, incremental_manual_examples=None):
        ex_queries = WikiSQLPrompting.manual_examples()
        prompt = """
REL is a database management system language that is similar to datalog.
Below are few examples of SQL code and REL code pairs that perform the same query.
Note that in REL queries, all strings should be lower case and all numbers and floating point numbers.
For example, string "Yale University" should be "yale university", and a single integer 1956 should be float 1956.0
Mathematical expressions are strings. Special characters such as "%" needs to be escaped with backslash as "\%"

Examples:
"""
        for i, ex in enumerate(ex_queries):
            idx = ex["idx"]
            sql = self.queries[idx]['query']
            rel = ex['query']
            prompt += f"""
{i + 1}.SQL: 
```
{sql}
```
"""
            prompt += f"""
{i + 1}.REL: 
```
{rel}
```
"""
        return prompt

    def proto_synthesis(self, n=100, split="train", verbose_level=1):
        chatgpt = ChatGPT()
        prompt = self.make_prototype_prompt()
        synths = []
        for samp in self.n_random_samples(n, split):
            sql = samp["query"]
            instruction = f"""
{prompt}

Given examples above, translate the following SQL query to REL. No explanation is needed.
Output only the REL code.
SQL:
```
{sql}
```
REL:
"""
            if verbose_level > 0:
                print("========== BEGIN ==============")
            if verbose_level > 1:
                print(instruction)
            elif verbose_level > 0:
                print(sql)
            answer = chatgpt.query_chat_gpt(instruction)
            if verbose_level > 0:
                print(answer)
            stripped_answer = answer.strip("REL:").strip().strip('`').strip()
            synth = {"idx": samp["idx"],
                     "query": stripped_answer}
            synths.append(synth)
            if verbose_level > 0:
                print("============ END ==============")
        out_path = self.cache_path / "chat_gpt_proto_synthesis.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            st = json.dumps(synths, indent=2)
            f.write(st)

    def proto_synthesis_one_batch(self, batch, num_results=1, verbose_level=1, incremental_manual_examples= None, **kwargs):
        """
        More efficient synthesis that reuses the prompt for multiple generations
        :param batch:
        :return:
        """
        chatgpt = ChatGPT()
        prompt = self.make_prototype_prompt(incremental_manual_examples)
        instruction = f"""
{prompt}

Given examples above, translate the following SQL queries to REL programs. No explanation is needed.
Output only the REL code one by one numbered with '1.REL:', '2.REL:', for example.
    """
        for idx, samp in enumerate(batch):
            sql = samp["query"]
            instruction += f"""
{idx + 1}.SQL:
```
{sql}
```
"""
        if verbose_level > 1:
            print(instruction)

        all_synths = []
        raw_answers = chatgpt.query_chat_gpt(instruction, num_results=num_results, **kwargs)
        if num_results == 1:
            raw_answers = [raw_answers]
        for raw_answer in raw_answers:
            synths = []
            try:
                answers = re.split("\d+.REL:", raw_answer)
                answers = answers[-len(batch):]
                if len(answers) != len(batch):
                    raise RuntimeError
                for samp, answer in zip(batch, answers):
                    if verbose_level > 0:
                        print("========== BEGIN ==============")
                    stripped_answer = answer.strip("REL:").strip().strip('`').strip()
                    synth = {"idx": samp["idx"],
                             "query": stripped_answer}
                    synths.append(synth)
                    if verbose_level > 0:
                        print(samp['query'])
                    if verbose_level > 0:
                        print(stripped_answer)
                    if verbose_level > 0:
                        print("============ END ==============")
            finally:
                all_synths.append(synths)
        if num_results == 1:
            return raw_answers[0], all_synths[0]
        else:
            return raw_answers, all_synths

    async def proto_synthesis_one_batch_async(self, batch, batch_number, split, verbose_level=1):
        """
        More efficient synthesis that reuses the prompt for multiple generations
        :param batch:
        :return:
        """
        chatgpt = ChatGPT()
        prompt = self.make_prototype_prompt()
        synths = []
        instruction = f"""
{prompt}

Given examples above, translate the following SQL queries to REL programs. No explanation is needed.
Output only the REL code one by one numbered with '1.REL:', '2.REL:', for example.
"""
        for idx, samp in enumerate(batch):
            sql = samp["query"]
            instruction += f"""
{idx + 1}.SQL:
```
{sql}
```
"""
        if verbose_level > 1:
            print(instruction)

        batch_answer = await chatgpt.query_chat_gpt_async(instruction)
        # answers = re.split("\d+.\s+REL:", batch_answer)
        # answers = answers[-10:]
        answers = self.split_chatgpt_answer(batch_answer)
        # if len(answers) != 10:
        #     raise RuntimeError
        for samp, answer in zip(batch, answers):
            if verbose_level > 0:
                print("========== BEGIN ==============")
            stripped_answer = answer.strip("REL:").strip().strip('`').strip()
            synth = {"idx": samp["idx"],
                     "query": stripped_answer}
            synths.append(synth)
            if verbose_level > 0:
                print(samp["query"])
            if verbose_level > 0:
                print(stripped_answer)
            if verbose_level > 0:
                print("============ END ==============")
        par_path = self.cache_path / f"chat_gpt_proto_synthesis_batch_raw_{split}"
        par_path.mkdir(parents=True, exist_ok=True)
        out_path = self.cache_path / f"chat_gpt_proto_synthesis_batch_raw_{split}/batch_{batch_number}.json"
        with open(out_path, 'w') as f:
            st = json.dumps((batch_answer, synths), indent=2)
            f.write(st)

    def proto_synthesis_multiple_batches(self, n, split="train", verbose_level=1):
        if n == -1:
            all_samples = self.all_samples(split)
            n = len(all_samples)
        else:
            all_samples = self.n_random_samples(n, split)

        batch_size = 10
        par = self.cache_path / f"chat_gpt_proto_synthesis_batch_raw_{split}"
        par.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(0, n, batch_size)):
            out_path = self.cache_path / f"chat_gpt_proto_synthesis_batch_raw_{split}/batch_{i}.json"
            if not out_path.exists():
                samples = all_samples[i: i + batch_size]
                batch_answer, syn = self.proto_synthesis_one_batch(samples, verbose_level)
                with open(out_path, 'w') as f:
                    st = json.dumps((batch_answer, syn), indent=2)
                    f.write(st)

    async def proto_synthesis_multiple_batches_async(self, n, split="train", verbose_level=0):
        if n == -1:
            all_samples = self.all_samples(split)
            n = len(all_samples)
        else:
            all_samples = self.n_random_samples(n, split)

        batch_size = 10
        par = self.cache_path / f"chat_gpt_proto_synthesis_batch_raw_{split}"
        par.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                tasks = []
                for i in tqdm(range(0, n, batch_size)):
                    out_path = self.cache_path / f"chat_gpt_proto_synthesis_batch_raw_{split}/batch_{i}.json"
                    if not out_path.exists():
                        samples = all_samples[i: i + batch_size]
                        task = asyncio.create_task(
                            self.proto_synthesis_one_batch_async(samples, i, split, verbose_level))
                        tasks.append(task)
                        if len(asyncio.all_tasks()) > 10:
                            await asyncio.gather(*tasks)
                await asyncio.gather(*tasks)
                break
            except (openai.error.RateLimitError, openai.error.APIError) as e:
                time.sleep(5)
                pass

        # 3000 tokens per request, 90000 tokens per minute
        # 30 requests per minute, 2 seconds per request

    def proto_synthesis_multiple_batches_run_async(self, n, split="train", verbose_level=0):
        asyncio.run(self.proto_synthesis_multiple_batches_async(n, split, verbose_level))

    def load_last_proto_synthesis(self):
        out_path = self.cache_path / "chat_gpt_proto_synthesis.json"
        with open(out_path, 'r') as f:
            st = f.read()
            synths = json.loads(st)
        return synths

    def split_chatgpt_answer(self, batch_answer):
        answers = re.split("\d+.\s*REL:", batch_answer)
        if answers[0].strip() == "":
            answers = answers[1:]
        answers = [answer.strip("REL:").strip().strip('`').strip() for answer in answers]
        return answers

    def load_all_chat_gpt_answer_batches(self):
        ret = {}
        for split in "train", "dev", "test":
            all_samples = self.all_samples(split)
            n = len(all_samples)
            batch_size = 10
            raw = []
            ss = []
            for i in tqdm(range(0, n, batch_size)):
                out_path = self.cache_path / f"chat_gpt_proto_synthesis_batch_raw_{split}/batch_{i}.json"
                with out_path.open('r') as f:
                    st = f.read()
                    batch_answer, synths = json.loads(st)
                    raw.append(batch_answer)
                    ss.extend(synths)

                # verify property
                # s = o[1]
                # if not (isinstance(s, list) and len(s) == 10):
                #     print(i)
            ret[split] = (raw, ss)

        # pick up, collate the batches and compute accuracy, dump back to disk
        # then we are done?
        return ret

    def n_random_samples(self, n=10, split="train"):
        queries, schemas, tables = load_wikisql(split)
        total = len(queries)

        relx = RELX()
        relx.main_create_engine_database()
        prompt_indices = [q['idx'] for q in WikiSQLPrompting.manual_examples()]

        # without replacement
        random.seed(43)
        success = 0
        samples = set()
        while True:
            chosen = random.sample(range(total), n)
            chosen = set(chosen).difference(set(prompt_indices)).difference(samples)
            success += len(chosen)
            samples.update(chosen)
            n = n - len(chosen)
            if n == 0:
                break

        sampled_queries = []
        for s in sorted(list(samples)):
            q = queries[s]
            sampled_queries.append(q)
        return sampled_queries

    def all_samples(self, split="train"):
        queries, schemas, tables = load_wikisql(split)
        return queries

    def get_parallel_actor_class(self):
        return ParallelWikiSQLPromptingActor

    def query_accuracy_parallel(self, split, num_workers=20, synthesized_queries=None, debug=False):
        """
        Split the synthesized queries to num_workers batches
        Make num_workers servers
        Query independently
        Collate the results in the end

        :param num_workers:
        :param synthesized_queries:
        :param verbose:
        :return:
        """
        if debug:
            chunk_length = 10
        else:
            chunk_length = len(synthesized_queries) // num_workers + 1

        splitted_queries = []
        for w in range(num_workers):
            splitted_queries.append(synthesized_queries[w * chunk_length:(w + 1) * chunk_length])

        workers = [self.get_parallel_actor_class().remote(split, rank, num_workers, ) for rank in range(num_workers)]

        promises = []
        for w in range(num_workers):
            queries = splitted_queries[w]
            worker = workers[w]
            promise = worker.query_accuracy.remote(queries)
            promises.append(promise)

        results = ray.get(promises)
        dataset = sum([r[0] for r in results], [])

        results = []
        for worker in workers:
            res = worker.shutdown_engine.remote()
            results.append(res)
        ray.get(results)

        return dataset

    def query_accuracy(self, synthesized_queries=None, verbose=True, pbar=True):
        correct = 0
        wrong = 0
        if synthesized_queries is None:
            synthesized_queries = self.load_last_proto_synthesis()

        dataset = []
        if self.relx is None:
            self.make_relx()
        for query in tqdm(synthesized_queries, disable=not pbar):
            idx = query['idx']
            error = False
            retry = 3
            while True:
                q = query['query']
                query_dict = self.queries[idx]
                sql_query = query_dict['query']
                table_id = query_dict['table_id']
                gold = query_dict['gold'][0]
                try:
                    json_string, csv_string = get_json_csv_string(self.tables[table_id])
                    schema = self.relx.to_rel_schema(self.schemas[table_id])
                    upload_csv(csv_string, table_id, schema, self.relx)
                    relation_name = table_id_to_rai_relation_name(table_id, "csv")

                    if verbose:
                        print("=======================")
                        print(f'Executing query {idx} on relation {relation_name}')
                        print(f"SQL query {sql_query}")
                        print(f"REL query {q}")
                        print(f"Gold result {gold}")
                    ret = self.relx.execute_rel_query_against_table(q, table_id)
                    break
                except:
                    retry -= 1
                    if retry == 0:
                        ret = "ERROR"
                        error = True
                        break
            if ret == gold:
                correct += 1
            else:
                wrong += 1
            if verbose:
                print(f"REL result {ret}")
                print("=======================")

            dat = {
                "idx": idx,
                "sql": sql_query,
                "rel": q,
                "gold": gold,
                "rel result": ret,
                "label": ret == gold,
                "error": error
            }
            dataset.append(dat)
        acc = correct / (correct + wrong)
        if verbose:
            print(f"accuracy {acc}")

        return dataset, acc


@ray.remote
class ParallelWikiSQLPromptingActor:

    def __init__(self, split, rank, total_workers):
        self.rank = rank
        self.total_workers = total_workers
        self.wsp = WikiSQLPrompting(engine_postfix=f"{rank}")
        self.wsp.set_split(split)

    def query_accuracy(self, synthesized_queries):
        dataset, acc = self.wsp.query_accuracy(synthesized_queries, verbose=False,
                                               pbar=self.rank == self.total_workers - 1)
        return dataset, acc

    def shutdown_engine(self):
        self.wsp.relx.shutdown_engine()


def synthesize_wikisql_from_chatgpt(wsp):
    splits = 'train', 'dev', 'test'
    # splits = 'dev', 'test'
    for split in splits:
        wsp.proto_synthesis_multiple_batches_run_async(-1, split)


def run_rel_queries_obtain_accuracy():
    wsp = WikiSQLPrompting()
    split_raw = wsp.load_all_chat_gpt_answer_batches()
    for split, dat in split_raw.items():
        dataset = wsp.query_accuracy_parallel(split, num_workers=20, synthesized_queries=dat[1])
        wsp.save_sql_rel_synth_dataset(dataset, split)


def re_run_rel_queries_obtain_accuracy():
    wsp = WikiSQLPrompting()
    split_raw = wsp.load_all_chat_gpt_answer_batches()
    for split in 'dev', 'test':
        dat = split_raw[split]
        dataset = wsp.query_accuracy_parallel(split, num_workers=20, synthesized_queries=dat[1])
        wsp.save_sql_rel_synth_dataset(dataset, split)


def main():
    # interactive_development(10)
    # ttest_execution_rel_against_tables()
    wsp = WikiSQLPrompting()
    # synthesize_wikisql_from_chatgpt(wsp)
    # samples = wsp.n_random_samples(100)
    # wsp.proto_synthesis() # do not use
    split_raw = wsp.load_all_chat_gpt_answer_batches()
    for split, dat in split_raw.items():
        # dataset, acc = wsp.query_accuracy(dat[1], verbose=False)
        dataset = wsp.query_accuracy_parallel(split, num_workers=20, synthesized_queries=dat[1])
        wsp.save_sql_rel_synth_dataset(dataset, split)

    # wsp.query_accuracy()


def get_prompt(n, split="train", verbose_level=1):
    wsp = WikiSQLPrompting()
    wsp.set_split(split)
    if n == -1:
        all_samples = wsp.all_samples(split)
        n = len(all_samples)
    else:
        all_samples = wsp.n_random_samples(n, split)

    batch_size = 10
    par = project_root() / f"cache/chat_gpt_proto_synthesis_batch_raw_{split}"
    par.mkdir(parents=True, exist_ok=True)
    chatgpt = ChatGPT()
    for i in tqdm(range(0, n, batch_size)):
        samples = all_samples[i: i + batch_size]
        prompt = wsp.make_prototype_prompt()
        instruction = f"""
        {prompt}

        Given examples above, translate the following SQL queries to REL programs. No explanation is needed.
        Output only the REL code one by one numbered with '1.REL:', '2.REL:', for example.
            """
        for idx, samp in enumerate(samples):
            sql = samp["query"]
            instruction += f"""
        {idx + 1}.SQL:
        ```
        {sql}
        ```
        """
        print(instruction)
        answer = chatgpt.query_chat_gpt(instruction)
        print(answer)


if __name__ == '__main__':
    get_prompt(-1)
    wsp = WikiSQLPrompting()
    synthesize_wikisql_from_chatgpt(wsp)
    run_rel_queries_obtain_accuracy()
