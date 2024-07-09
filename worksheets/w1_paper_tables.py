# section 3
import random

from sql.export import KaggleSqlRelSource
from tqdm import tqdm
import pandas

from sql.synth import SynthesisPipeline


# today:
def table_two():

    # load sql dataset
    # break down the sql components
    # calculate the percentage of validity
    
    pipeline = SynthesisPipeline(incremental_manual_examples=None)
    step = 5
    # 1
    pipeline.kaggle.download_datasets(step)
    # 2
    pipeline.get_all_schemas_and_top_rows(step)
    # 3
    pipeline.chatgpt_synthesize_sql_queries(step)
    # 4
    pipeline.filter_sql_execution_with_ground_truth(step)
    
    source = KaggleSqlRelSource()
    flags = pipeline.sql_executable_flag

    sql_programs_generated = 0
    
    all_count = 0
    all_correct = 0
    all_error = 0
    all_empty = 0
    cls_count = {k: 0 for k in source.sql_keywords.values()}
    cls_correct = {k: 0 for k in source.sql_keywords.values()}
    cls_empty = {k: 0 for k in source.sql_keywords.values()}
    cls_error = {k: 0 for k in source.sql_keywords.values()}

    # print_cnt = 100
    synthesized_sqls =  pipeline.synthesized_sql
    flags = pipeline.sql_executable_flag
    # for rowid, row in tqdm(enumerate(source.catalog.iloc), total=len(source.catalog)):
    #     sqls, sql_executable = row['sql_translated'], row['sql_executable']
    for sqls, flag in zip(synthesized_sqls, flags):
        if sqls is not None and flag is not None:
            sql_programs_generated += len(sqls)
            for sql, executable in zip(sqls, flag):
                clss = source.classify_sql(sql[1])
                for c in clss:
                    # if c == "group by" and print_cnt >0:
                    #     print_cnt -= 1
                    #     print(sql)
                    cls_count[c] += 1
                    if executable=="Valid":
                        cls_correct[c] += 1
                    elif executable == "Error":
                        cls_error[c]+=1
                    else:
                        assert executable == "Empty"
                        cls_empty[c] +=1
                if executable =="Valid":
                    all_correct += 1
                elif executable == "Error":
                    all_error +=1
                else:
                    all_empty +=1
                all_count += 1

    cls_correct_percent = {}
    cls_error_percent = {}
    cls_empty_percent = {}

    for k in cls_count:
        cls_correct_percent[k] = cls_correct[k] / cls_count[k]
        cls_error_percent[k] = cls_error[k] / cls_count[k]
        cls_empty_percent[k] = cls_empty[k] / cls_count[k]

    dat = {
        'type': [],
        'sql count': [],
        "sql correct": [],
        'sql correct percentage': [],
        'sql error percentage':[],
        'sql empty percentage': []
    }
    for k in source.sql_keywords.values():
        dat['type'].append(k)
        dat['sql count'].append(cls_count[k])
        dat['sql correct'].append(cls_correct[k])
        dat['sql correct percentage'].append(cls_correct_percent[k])
        dat['sql empty percentage'].append(cls_empty_percent[k])
        dat['sql error percentage'].append(cls_error_percent[k])

    df = pandas.DataFrame(dat)
    df = df.sort_values(by="sql count", ascending=False)
    df.reset_index(drop=True, inplace=True)
    df['sql correct percentage'] = df['sql correct percentage'] * 100
    df['sql empty percentage'] = df['sql empty percentage'] * 100
    df['sql error percentage'] = df['sql error percentage'] * 100

    df['type'] = df['type'].apply(lambda x: f"\\texttt{{{x.upper()}}}")


    df['sql correct percentage'] = df['sql correct percentage'].round(2)

    df = df.rename(columns = {
        "type": "SQL Component",
        'sql count':"SQL Count",
        'sql correct': "Valid SQL",
        'sql correct percentage': "\% Valid",
        'sql empty percentage': "\% Empty",
        'sql error percentage': "\% Error"
    })

    df2 = pandas.DataFrame([["Total", all_count, all_correct, all_correct/all_count*100, all_empty/all_count*100,
                             all_error/all_count*100]], columns=df.columns)
    df = pandas.concat([df2, df, ])
    tex = df.to_latex(index=False, float_format="%.2f")

    print(tex)


def features_statistics_examples_examples_sql():
    source = KaggleSqlRelSource()
    sql_programs_generated = 0
    pick = 10
    table = {"SQL Annotation":[],
             "English Annotation":[],
             "Execution Validity":[],
             "idx":[]}
    valid_cnt= 100
    invalid_cnt = 100
    for valid in (True, False):
        if valid:
            remaining = valid_cnt
        else:
            remaining = invalid_cnt
        while remaining> 0:
            rowidx  = random.randint(0, len(source.catalog)-1)
            row = source.catalog.iloc[rowidx]
            sqls, sql_executable = row['sql_translated'], row['sql_executable']
            if sqls is not None and sql_executable is not None:
                idx  = random.randint(0, len(sqls)-1)
                if sql_executable[idx]!=valid:
                    continue
                table["SQL Annotation"].append(sqls[idx][1])
                table["English Annotation"].append(sqls[idx][0])
                table["Execution Validity"].append(sql_executable[idx])
                table["idx"].append((rowidx, idx))
                if valid:
                    valid_cnt -=1
                    remaining = valid_cnt
                else:
                    invalid_cnt -=1
                    remaining = invalid_cnt
    df = pandas.DataFrame(table)


    valid_indices = [(49823, 4), (4207, 12), (20379, 3) ]
    invalid_indices = [(26100, 2), (41750, 2), (41984,0)]

    table = {
             "English and SQL Annotation":[],
             "Valid":[]}
    
    t =0
    for indices in valid_indices, invalid_indices:
        for rowidx, idx in indices:
            row = source.catalog.iloc[rowidx]
            sqls, sql_executable = row['sql_translated'], row['sql_executable']
            if sqls is not None and sql_executable is not None:
                annotation = sqls[idx][0] + "\\newline" + "\\texttt{"+sqls[idx][1].replace("_","\_")+ "}"
                table["English and SQL Annotation"].append(annotation)
                table["Executable"].append(not bool(t))
        t+=1
    df = pandas.DataFrame(table)

    print(df.to_latex(index=False))

    table = {
        "English Annotation": [],
        "SQL Annotation": [],
        "Valid": []
    }

    t = 0
    for indices in valid_indices, invalid_indices:
        for rowidx, idx in indices:
            row = source.catalog.iloc[rowidx]
            sqls, sql_executable = row['sql_translated'], row['sql_executable']
            if sqls is not None and sql_executable is not None:
                # annotation = sqls[idx][0] + "\\newline" + "\\texttt{" + sqls[idx][1].replace("_", "\_") + "}"
                table["English Annotation"].append(sqls[idx][0] )
                table["SQL Annotation"].append("\\texttt{" + sqls[idx][1].replace("_", "\_") + "}")
                table["Executable"].append(not bool(t))
        t += 1
    df = pandas.DataFrame(table)

    print(df.to_latex(index=False))

    print("DONE")

# section 4
def table_three_rel_accuracy():
    source = KaggleSqlRelSource()

    all_count = 0
    all_correct = 0
    all_error = 0
    all_unequal =0
    cls_count = {k: 0 for k in source.sql_keywords.values()}
    cls_correct = {k: 0 for k in source.sql_keywords.values()}
    cls_error = {k: 0 for k in source.sql_keywords.values()}
    cls_unequal= {k: 0 for k in source.sql_keywords.values()}

    for row in tqdm(source.sql_rel.iloc, total=len(source.sql_rel)):
        rels, rels_correct = row['rel_code'], row['rel_executable']
        sqls = row['sql_code']
        infos = row['rel_executable_info']
        if rels is not None and rels_correct is not None:
            for rel, correct, sql, info in zip(rels, rels_correct, sqls, infos):
                clss = source.classify_sql(sql[1])
                for c in clss:
                    cls_count[c] += 1
                    if correct:
                        cls_correct[c] += 1
                    elif info == "error":
                        cls_error[c] += 1
                    elif info == "comparison":
                        cls_unequal[c] +=1
                if info == "correct":
                    all_correct += 1
                elif info == "error":
                    all_error +=1
                elif info == "comparison":
                    all_unequal +=1
                all_count += 1
    cls_correct_percent = {}
    cls_error_percent = {}
    cls_unequal_percent = {}

    for k in cls_count:
        cls_correct_percent[k] = cls_correct[k] / cls_count[k]
        cls_error_percent[k] = cls_error[k] / cls_count[k]
        cls_unequal_percent[k] = cls_unequal[k] / cls_count[k]

    dat = {
        'type': [],
        'sql count': [],
        "rel correct": [],
        'rel correct percentage': [],
        "\\% Rel Error":[],
        "\\% Different Results":[]
    }
    for k in source.sql_keywords.values():
        dat['type'].append(k)
        dat['sql count'].append(cls_count[k])
        dat['rel correct'].append(cls_correct[k])
        dat['rel correct percentage'].append(cls_correct_percent[k])
        dat['\\% Rel Error'].append(cls_error_percent[k])
        dat['\\% Different Results'].append(cls_unequal_percent[k])

    df = pandas.DataFrame(dat)
    df = df.sort_values(by="sql count", ascending=False)
    df.reset_index(drop=True, inplace=True)

    ex_demo = {k: 0 for k in source.sql_keywords.values()}
    for e in source.manual_examples.iloc:
        sql = e.sql
        clss = source.classify_sql(sql)
        for c in clss:
            ex_demo[c]+=1
    exdf = {"type": [],
            "Examples": []}
    for k,v in ex_demo.items():
        exdf['type'].append(k)
        exdf['Examples'].append(v)
    exdf = pandas.DataFrame(exdf)
    df = df. merge(exdf, on="type", how="outer")
    
    df['type'] = df['type'].apply(lambda x: f"\\texttt{{{x.upper()}}}")
    df['rel correct percentage'] = df['rel correct percentage'] * 100
    df['\\% Rel Error'] = df['\\% Rel Error'] * 100
    df['\\% Different Results'] = df['\\% Different Results'] * 100

    df = df.rename(columns={
        "type": "SQL Component",
        'sql count': "SQL Translated",
        'rel correct': "Rel Correct",
        'rel correct percentage': "\\% Rel Correct"
    })


    df2 = pandas.DataFrame([["Total", all_count, all_correct, all_correct / all_count * 100, all_error/all_count *100,
                             all_unequal/all_count*100, len(source.manual_examples)]], columns=df.columns)
    df = pandas.concat([df2, df, ])

    df = df [['SQL Component', 'Examples', "SQL Translated", 'Rel Correct', '\\% Rel Correct', "\\% Rel Error", '\\% Different Results']]

    tex = df.to_latex(index=False, float_format="%.2f")

    print(tex)


def sql_to_rel_examples():
    source = KaggleSqlRelSource()
    sql_programs_generated = 0
    pick = 10
    table = {"SQL":[],
             "Rel":[],
             "Ex. Acc.":[],
             "idx":[]}

    types = {
        "correct":100,
        "error":100,
        "comparison":100,
    }
    for type in types:
        remaining = types[type]
        while remaining> 0:
            rowidx  = random.randrange(0, len(source.sql_rel))
            row = source.sql_rel.iloc[rowidx]
            sql_code, rel_code, rel_executable_info = row['sql_code'], row['rel_code'], row ['rel_executable_info']
            if sql_code is not None and rel_code is not None and rel_executable_info is not None:
                idx = random.randrange(0, len(sql_code))
                if rel_executable_info[idx]!=type:
                    continue
                table["SQL"].append(sql_code[idx][1])
                table["Rel"].append(rel_code[idx])
                table["Ex. Acc."].append(type)
                table["idx"].append((rowidx, idx))
                types[type] -=1
                remaining = types[type]

    df = pandas.DataFrame(table)

    indices = {
        "Correct":[(1501, 3), (6232, 8), (25094, 2)],
        "Error":[(26349, 8), (22195, 6), (3458, 6)],
        "Comparison":[(10134, 6), (13065, 6), (20784, 1)]
    }

    table = {
        "SQL": [],
        "Rel Translated": [],
        "Ex. Acc.": []
    }


    for type, indi in indices.items():
        for rowidx, idx in indi:
            row = source.sql_rel.iloc[rowidx]
            sql_code, rel_code, rel_executable_info = row['sql_code'], row['rel_code'], row ['rel_executable_info']
            sql = sql_code[idx]
            sql_code = "\\texttt{"+sql[1].replace("_","\_")+ "}"
            rel_code = "\\texttt{"+rel_code[idx].replace("_","\_").replace("\n", "\\newline ")+ "}"
            
            table['SQL'].append(sql_code)
            table['Rel Translated'].append(rel_code)
            table['Ex. Acc.'].append(type)
    df = pandas.DataFrame(table)

    print(df.to_latex(index=False))


    print("DONE")



def performance_convergence_incremental_prompt_engineering():
    pass


# section 5 is in the tabpfn repo and R

if __name__ == '__main__':
    table_two()
    # features_statistics_examples_examples_sql()
    # table_three_rel_accuracy()