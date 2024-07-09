"""
We need three types of SQL execution results.

We also need to confirm that the results are the same as the last time.
Otherwise, alter the new table to match.
"""


from sql.synth import KagglePrompting, SynthesisPipeline


def three_types():
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
    
    flags = pipeline.sql_executable_flag
    cnt = 0
    ok_db_count = 0

    for f in flags:
        if f is not None:
            db_ok = False
            for v in f:
                if v == "Valid":
                    cnt +=1
                    db_ok=True
            if db_ok:
                ok_db_count+=1
    print(cnt)
    print(ok_db_count)
    print("DONE")


def three_types_break_down():
    pass
    
if __name__ == '__main__':
    three_types()