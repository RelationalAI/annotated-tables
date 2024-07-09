import itertools

import pandas
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def make_table():
    a = {
        # "dataset": ["AutoML-CC18", "AnnotatedTables"],
        "time budget": ["1 Min", "5 Mins"],
        "model": ["transformer", "autogluon", "diff"],
    }
    source_60 = pandas.read_csv(f"./dump_results_{60}.csv")
    source_300 = pandas.read_csv(f"./dump_results_{300}.csv")

    columns = list(itertools.product(*a.values()))

    metrics = ["mean AUROC OVO", "median AUROC OVO", "stdv AUROC OVO", "mean CE", "median CE", "stdv CE", "mean time"]
    df = {"metric": metrics}
    df.update({c: [] for c in columns})

    for time in a['time budget']:
        if time == "1 Min":
            source = source_60
        else:
            source = source_300
        source["diff_roc"] = source["transformer_roc"] - source['autogluon_roc']
        source["diff_cross_entropy"] = source["transformer_cross_entropy"] - source['autogluon_cross_entropy']
        source["diff_time"] = source["transformer_time"] - source['autogluon_time']

        columns = source.columns

        for metric in metrics:
            for model in a['model']:
                cols = [c for c in columns if model in c]
                if "AUROC" in metric:
                    col = [c for c in cols if "roc" in c]
                elif "CE" in metric:
                    col = [c for c in cols if "cross_entropy" in c]
                else:
                    assert "time" in metric
                    col = [c for c in cols if "time" in c]
                assert len(col) == 1, col
                col = col[0]
                co = source[col]
                if "mean" in metric:
                    val = co.mean()
                elif "median" in metric:
                    val = co.median()
                elif "stdv" in metric:
                    val = co.std()
                else:
                    raise

                df[(time, model)].append(val)
    df = pandas.DataFrame(df)
    print(df)
    print(df.to_latex(index=False, float_format="%.3f", escape=False))
    print(len(source_60))
    print(len(source_300))


if __name__ == '__main__':
    make_table()
