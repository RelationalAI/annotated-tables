import pickle
from pathlib import Path

import pandas
from tqdm import tqdm

from run_kaggle import RunKaggle


class Correlation:
    def __init__(self, max_time):
        self.ka = RunKaggle(max_time=max_time)

    def dump_results(self, global_results=None, dataset_names=None, postfix="", factors=True):
        if global_results is None:
            assert dataset_names is None
            with open(f"./global_results_{self.ka.max_time}.pkl", 'rb') as f:
                global_results, dataset_names = pickle.load(f)
        else:
            assert dataset_names is not None

        df, global_results, metrics = self.ka.load_models_analyze_results(global_results, dataset_names)
        performance = {
            metric_name: pandas.DataFrame(metric[2]) for metric_name, metric in metrics.items()
        }

        output = []

        adapter = self.ka.proto.adapter
        catalog = self.ka.proto.adapter.exported_kaggle.catalog.set_index('ref')
        for name in tqdm(dataset_names):
            perfs = {}
            for metric, df in performance.items():
                perf = df.loc[name]
                for model, value in perf.items():
                    perfs[model + '_' + metric] = value
            ref = name.replace(':', "/")
            output_row = {
                "name": ref
            }
            output_row.update(perfs)

            if factors:
                catalog_row = catalog.loc[ref]
                schema = catalog_row['schemas'][0]
                suggested_columns = self.ka.proto.suggested_columns[ref]
                data = adapter.get_table_data(ref)
                table, input_columns, output_column = self.ka.input_output_suitable_for_tabpfn(schema,
                                                                                               suggested_columns,
                                                                                               data, catalog_row)
                stats = self.find_factors(table, input_columns, output_column)
                output_row.update(stats)
            output.append(output_row)
        output = pandas.DataFrame(output)
        output.to_csv(f"./dump_results_{self.ka.max_time}{postfix}.csv")
        print(f"Dumped to ./dump_results_{self.ka.max_time}{postfix}.csv")
        return output

    def find_factors(self, table, input_columns, output_column):
        # number of columns
        # number of rows
        # missingness
        # number of classes of the target column

        number_of_input_columns = len(input_columns)
        number_of_rows = len(table)
        missingness = table.isnull().values.any()
        number_of_classes = table[output_column].nunique()
        factors = {
            "number_of_input_columns": number_of_input_columns,
            "number_of_rows": number_of_rows,
            "missingness": missingness,
            "number_of_classes": number_of_classes
        }
        return factors


class Inspect:
    # what is the model's input?
    # what is the model's output?
    # is there truncation of the number of rows and number of columns? yes on the rows,
    # why do the AUC add to one? How is AUC calculated? Something fishy.

    def __init__(self):
        self.ka = RunKaggle()

        catalog = self.ka.proto.adapter.exported_kaggle.catalog.copy()
        catalog['index'] = catalog['ref']
        catalog = catalog.set_index('index')
        self.catalog = catalog[~catalog.index.duplicated(keep='first')]

    def inspect_one(self, name):
        catalog = self.catalog

        ref = name.replace(':', "/")
        row = catalog.loc[ref]
        schema = row['schemas'][0]

        suggested_columns = self.ka.proto.suggested_columns[name]
        data = self.ka.proto.adapter.get_table_data(name)
        suitable = self.ka.input_output_suitable_for_tabpfn(
            schema, suggested_columns, data, row
        )
        table, input_columns, output_column = suitable
        six_tuple_dataset = self.ka.convert_dataset_to_six_tuple(
            schema, input_columns, output_column, table, row
        )
        res = self.ka.tabpfn_evaluate(six_tuple_dataset, overwrite=True, save=False)
        return res


def main():
    cor = Correlation()
    cor.dump_results()

