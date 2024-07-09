import json
import traceback

import autogluon
import numpy as np
import pandas
import pandas as pd
from autogluon.core.utils import infer_problem_type
from scipy.stats import rankdata

from aproto import FeasibilityPrototype, AutoG, kaggle_path, SQLCacher
from analyze_one_dataset import main
import torch
import pandas
from scripts import tabular_metrics
from analyze_one_dataset import eval_method
from tqdm import tqdm

from scripts.tabular_evaluation import TabPFNError
from scripts import tabular_baselines
from tabpfn.scripts.tabular_metrics import time_metric, make_metric_matrix
from aproto import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



def calculate_score(metric, name, global_results, ds, eval_positions, aggregator='mean', limit_to='', ds_names=False):
    """
    Calls calculate_metrics_by_method with a range of methods. See arguments of that method.
    :param limit_to: This method will not get metric calculations.
    """
    for m in global_results:
        if limit_to not in m:
            continue
        calculate_score_per_method(metric, name, global_results[m], ds, eval_positions, aggregator=aggregator,
                                   ds_names=ds_names)


def calculate_score_per_method(metric, name: str, global_results: dict, ds: list, eval_positions: list,
                               aggregator: str = 'mean',
                               ds_names=False):
    """
    Calculates the metric given by 'metric' and saves it under 'name' in the 'global_results'

    :param metric: Metric function
    :param name: Name of metric in 'global_results'
    :param global_results: Dicrtonary containing the results for current method for a collection of datasets
    :param ds: Dataset to calculate metrics on, a list of dataset properties
    :param eval_positions: List of positions to calculate metrics on
    :param aggregator: Specifies way to aggregate results across evaluation positions
    :return:
    """
    aggregator_f = np.nanmean if aggregator == 'mean' else np.nansum
    if not ds_names:
        names = [d[0] for d in ds]
    else:
        names = ds
    for pos in eval_positions:
        valid_positions = 0

        for d in names:
            if f'{d}_outputs_at_{pos}' in global_results:
                preds = global_results[f'{d}_outputs_at_{pos}']
                y = global_results[f'{d}_ys_at_{pos}']

                preds, y = preds.detach().cpu().numpy() if torch.is_tensor(
                    preds) else preds, y.detach().cpu().numpy() if torch.is_tensor(y) else y

                try:
                    if metric == "time_metric":
                        global_results[f'{d}_{name}_at_{pos}'] = global_results[f'{d}_time_at_{pos}']
                        valid_positions = valid_positions + 1
                    else:
                        global_results[f'{d}_{name}_at_{pos}'] = aggregator_f(
                            [metric(y[split], preds[split]) for split in range(y.shape[0])])
                        valid_positions = valid_positions + 1
                except Exception as err:
                    print(f'Error calculating metric with {err}, {type(err)} at {d} {pos} {name}')
                    global_results[f'{d}_{name}_at_{pos}'] = np.nan
            else:
                global_results[f'{d}_{name}_at_{pos}'] = np.nan

        if valid_positions > 0:
            global_results[f'{aggregator}_{name}_at_{pos}'] = aggregator_f(
                [global_results[f'{d}_{name}_at_{pos}'] for d in names])
        else:
            global_results[f'{aggregator}_{name}_at_{pos}'] = np.nan

    for d in names:
        metrics = [global_results[f'{d}_{name}_at_{pos}'] for pos in eval_positions]
        metrics = [m for m in metrics if not np.isnan(m)]
        global_results[f'{d}_{aggregator}_{name}'] = aggregator_f(metrics) if len(metrics) > 0 else np.nan

    metrics = [global_results[f'{aggregator}_{name}_at_{pos}'] for pos in eval_positions]
    metrics = [m for m in metrics if not np.isnan(m)]
    global_results[f'{aggregator}_{name}'] = aggregator_f(metrics) if len(metrics) > 0 else np.nan


def make_metric_matrix(global_results, methods, pos, name, ds):
    result = []
    for m in global_results:
        try:
            result += [[global_results[m][d + '_' + name + '_at_' + str(pos)] for d in ds]]
        except Exception as e:
            # raise(e)
            result += [[np.nan]]
    result = np.array(result)
    result = pd.DataFrame(result.T, index=[d for d in ds], columns=[k for k in list(global_results.keys())])
    result = result[~result.index.duplicated(keep='first')]

    raw_matrix, matrix_stds, matrix_per_split = [], [], []

    for method in methods:
        raw_matrix += [result.iloc[:, [c.startswith(method + '_time') for c in result.columns]].mean(axis=1)]
        matrix_stds += [result.iloc[:, [c.startswith(method + '_time') for c in result.columns]].std(axis=1)]
        matrix_per_split += [result.iloc[:, [c.startswith(method + '_time') for c in result.columns]]]

    matrix_means = pd.DataFrame(raw_matrix, index=methods).T
    matrix_stds = pd.DataFrame(matrix_stds, index=methods).T
    raw_matrix_dict = {methods[i]: raw_matrix[i] for i in range(len(methods))}

    return matrix_means, matrix_stds, matrix_per_split, raw_matrix_dict


def make_ranks_and_wins_table(matrix, higher_the_better=True):
    # default: higher the better
    if not higher_the_better:
        coef = 1
    else:
        coef = -1
    for dss in matrix.T:
        matrix.loc[dss] = rankdata(coef * matrix.round(3).loc[dss])
    ranks_acc = matrix.mean()
    wins_acc = (matrix == 1).sum()

    return ranks_acc, wins_acc


class RunKaggle:
    def __init__(self, start=None, end=None, max_time=60):
        # this is sql code
        assert (start is None and end is None) or (start is not None and end is not None)
        self.proto = FeasibilityPrototype(start=start, end=end)
        self.analyze_one_dataset = ()
        self.autog = AutoG()
        self.eval_positions = [1000]
        self.max_time = max_time
        self.max_times = [max_time, ]  # [60, ]
        self.metric_used = tabular_metrics.auc_metric
        self.cacher = SQLCacher()
        self.methods = [
            'transformer', 'autogluon'
        ]

    def input_output_suitable_for_tabpfn(self, schema, suggested_columns, data, row):
        # input columns are float
        tbl = next(iter(data.values()))
        renamed = self.autog.sql.rename_columns(tbl.columns)
        tbl = tbl.rename(columns=renamed)
        # tbl = tbl.dropna()

        input_columns = suggested_columns["input_columns"]

        cleaned_input_columns = []
        for i in input_columns:
            try:
                if tbl.dtypes[i] in ("float64", "int64"):
                    cleaned_input_columns.append(i)
                # else:
                #     print(tbl.dtypes[i])
            except (KeyError, ValueError) as e:
                return None

        # output column is categorical
        try:
            output_column = suggested_columns["output_column"]
        except KeyError:
            return None
        try:
            if tbl.dtypes[output_column] != "object":
                return None
        except (KeyError, ValueError) as e:
            return None

        if len(cleaned_input_columns) < 2:
            return None

        try:
            ptype = infer_problem_type(tbl[output_column], silent=True)
        except ValueError:
            return None
        if ptype not in ('multiclass', 'binary'):
            print(ptype)
            return None

        return tbl, cleaned_input_columns, output_column

    def convert_dataset_to_six_tuple(
            self, schema, input_columns, output_column, df, row
    ):
        """
        the tabpfn repo uses a six tuple to represent each dataset
        name
        input_features: float tensor [row, features]
        label: int [row]
        empty list []
        name of features [features]
        config
        """

        name = row["ref"]
        name = name.replace("/", ":")

        inputs = df[input_columns]
        input_features = torch.tensor(inputs.values)
        feature_names = inputs.columns.tolist()

        label = df[output_column].astype('category')
        label = pandas.factorize(label)
        label = torch.tensor(label[0])
        config = {'samples_capped': False, 'classes_capped': False, 'feats_capped': False}
        return name, input_features, label, [], feature_names, config

    def tabpfn_evaluate(self, dataset, overwrite, load_only=False, **kwargs):
        eval_positions = self.eval_positions
        max_times = self.max_times
        metric_used = self.metric_used
        methods = self.methods
        task_type = 'multiclass'

        global_results = {}

        for method in methods:
            for max_time in max_times:
                ret = eval_method([dataset], task_type, method, eval_positions, max_time, metric_used,
                                  1, overwrite=overwrite, load_only=load_only, **kwargs)
                global_results[method + '_time_' + str(max_time) +
                               tabular_baselines.get_scoring_string(metric_used, usage='') +
                               '_split_' + str(1)] = ret
        return global_results

    def load_models_analyze_results(self, global_results, datasets):
        """
        TODO Develop this chunk to get results, then scale up
        """

        eval_positions = self.eval_positions
        max_times = self.max_times
        split_number = 1

        limit_to = ''
        calculate_score(tabular_metrics.auc_metric, 'roc', global_results, datasets, eval_positions + [-1],
                        limit_to=limit_to, ds_names=True)
        calculate_score(tabular_metrics.cross_entropy, 'cross_entropy', global_results, datasets,
                        eval_positions + [-1],
                        limit_to=limit_to, ds_names=True)
        calculate_score(tabular_metrics.accuracy_metric, 'acc', global_results, datasets, eval_positions + [-1])
        calculate_score("time_metric", 'time', global_results, datasets, eval_positions + [-1],
                        aggregator='sum', limit_to=limit_to, ds_names=True)
        calculate_score("time_metric", 'time', global_results, datasets, eval_positions + [-1],
                        aggregator='mean', limit_to=limit_to, ds_names=True)
        calculate_score(tabular_metrics.count_metric, 'count', global_results, datasets, eval_positions + [-1],
                        aggregator='sum', limit_to=limit_to, ds_names=True)

        df_ = []
        metric_keys = ['roc', 'cross_entropy', 'time']
        pos = str(self.eval_positions[0])
        max_time = max_times[0]
        global_results_filtered = {**global_results}
        global_results_filtered = {k: global_results_filtered[k] for k in global_results_filtered.keys() if
                                   '_time_' + str(max_time) +
                                   tabular_baselines.get_scoring_string(self.metric_used,
                                                                        usage='') + '_' in k or 'transformer' in k}

        time_matrix, _, _, _ = make_metric_matrix(global_results_filtered, self.methods, pos, 'time', datasets)
        time_matrix = time_matrix.mean()
        # this is fine.

        # Calculate ranks and wins per split
        ranks_and_wins = {k: {} for k in global_results}
        metrics = {}
        for metric_key in metric_keys:
            res, matrix = self.generate_ranks_and_wins_table(datasets, global_results_filtered, metric_key,
                                                             max_time,
                                                             split_number,
                                                             time_matrix)
            ranks, wins = res
            metrics[metric_key] = ranks, wins, matrix
            for method in self.methods:
                method_ = method + '_time_' + str(max_time) + tabular_baselines.get_scoring_string(
                    self.metric_used,
                    usage='')  # if method != 'transformer' else method
                global_results[method_ + '_split_' + str(split_number)][
                    'mean_rank_' + metric_key + f'_at_{pos}'] = \
                    ranks[method]
                global_results[method_ + '_split_' + str(split_number)][
                    'mean_wins_' + metric_key + f'_at_{pos}'] = \
                    wins[method]
                ranks_and_wins[method_ + '_split_' + str(split_number)][
                    'mean_rank_' + metric_key + f'_at_{pos}'] = \
                    ranks[method]
                ranks_and_wins[method_ + '_split_' + str(split_number)][
                    'mean_wins_' + metric_key + f'_at_{pos}'] = \
                    wins[method]

        # for method in global_results.keys():
        #    global_results[method]['mean_rank_'+metric_key+f'_at_{pos}'] = ranks[]

        avg_times = {}
        for method_ in self.methods:
            avg_times[method_] = []
            method = method_ + '_time_' + str(max_time) + tabular_baselines.get_scoring_string(
                self.metric_used,
                usage='') + '_split_' + str(
                split_number)
            avg_times[method_] += [global_results[method][f'mean_time_at_{pos}']]
        avg_times = pd.DataFrame(avg_times).mean()
        for metric_key in metric_keys:
            for ranking in ['', 'rank_', 'wins_']:
                for method_ in self.methods:
                    method = method_ + '_time_' + str(max_time) + tabular_baselines.get_scoring_string(
                        self.metric_used, usage='') + '_split_' + str(split_number)

                    # if global_results[method][f'sum_count_at_{pos}'] <= 29:
                    #     print('Warning not all datasets generated for ' + method + ' ' + str(
                    #         global_results[method][f'sum_count_at_{pos}']))

                    # time = global_results[method]['mean_time'] if ranking == '' else max_time
                    time = max_time
                    df_ += [{
                        # 'metric' + ranking + metric_key: global_results[method][
                        #     'mean_' + ranking + metric_key + f'_at_{pos}'],
                        "ranking": ranking if ranking != "" else "default",
                        "metric_key": metric_key,
                        "metric_ranking_value": global_results[method][
                            'mean_' + ranking + metric_key + f'_at_{pos}'],
                        'real_time': avg_times[method_],
                        'time': time,
                        'method': method_,
                        'split_number': split_number}]
                    # For Roc AUC Plots
                    # if 'transformer' in method:
                    #    df_ += [{'metric'+ranking+metric_key: global_results[method]['mean_'+ranking+metric_key+f'_at_{pos}'], 'real_time': avg_times[method_], 'time': time, 'method': method_, 'split_number': split_number}]
                    #    df_ += [{'metric'+ranking+metric_key: global_results[method]['mean_'+ranking+metric_key+f'_at_{pos}'], 'real_time': max(avg_times), 'time': max(max_times), 'method': method_, 'split_number': split_number}]

        df = pd.DataFrame(df_)
        return df, global_results, metrics

    def generate_ranks_and_wins_table(self, datasets, global_results_filtered, metric_key, max_time, split_number,
                                      time_matrix):
        global_results_filtered_split = {**global_results_filtered}
        global_results_filtered_split = {k: global_results_filtered_split[k] for k in
                                         global_results_filtered_split.keys()
                                         if '_time_' + str(max_time) + tabular_baselines.get_scoring_string(
                self.metric_used,
                usage='') + '_split_' + str(
                split_number) in k or 'transformer_split_' + str(split_number) in k}

        matrix, matrix_stds, _, raw_matrix = make_metric_matrix(global_results_filtered_split, self.methods,
                                                                str(self.eval_positions[0]), metric_key, datasets)
        for method in self.methods:
            if time_matrix[method] > max_time * 2:
                matrix[method] = np.nan
            # = np.nan

        if metric_key == 'cross_entropy':
            matrix = (matrix.fillna(100))
            higher_the_better = False
        elif metric_key == 'time':
            higher_the_better = False
        else:
            matrix = matrix.fillna(-1)
            higher_the_better = True
        return make_ranks_and_wins_table(matrix.copy(), higher_the_better=higher_the_better), raw_matrix

    def eval_starts(self, name, time):
        key = {
            "name": name,
            "time": time
        }
        key_st = json.dumps(key)
        res = self.cacher.load_cache(EvaluatedCache2, key_st)
        if res is None:
            self.cacher.write_cache(EvaluatedCache2, key_st, "failed")  # later over-written to be successful
        else:
            return res

    def prev_status(self, name, time):
        key = {
            "name": name,
            "time": time
        }
        key_st = json.dumps(key)
        res = self.cacher.load_cache(EvaluatedCache2, key_st)
        return res

    def eval_ends(self, name, time):
        key = {
            "name": name,
            "time": time
        }
        key_st = json.dumps(key)
        self.cacher.write_cache(EvaluatedCache2, key_st, "successful", overwrite=True)

    def run_all_evaluation(self, overwrite=True):
        # pipelined
        self.proto.suggest_input_target(load=True)

        for tupl in tqdm(self.proto.kaggle_iterator(), total=len(self.proto.selected_indices)):
            if tupl is None:
                continue
            schema, suggested_columns, data, row = tupl
            prev_status = self.eval_starts(row['ref'], self.max_times[0])
            try:
                suitable = self.input_output_suitable_for_tabpfn(
                    schema, suggested_columns, data, row
                )
                if suitable is None:
                    continue
                table, input_columns, output_column = suitable
                six_tuple_dataset = self.convert_dataset_to_six_tuple(
                    schema, input_columns, output_column, table, row
                )
                try:
                    self.tabpfn_evaluate(six_tuple_dataset, overwrite, False)
                except TabPFNError:
                    continue
                self.eval_ends(row['ref'], self.max_times[0])
            except Exception as e:
                traceback.print_exc()

    def analyze_and_save_results(self):
        # pipelined
        self.proto.suggest_input_target(load=True)

        global_results = {}
        dataset_names = []

        for tupl in tqdm(self.proto.kaggle_iterator(), total=len(self.proto.selected_indices)):
            if tupl is None:
                continue
            schema, suggested_columns, data, row = tupl
            prev_status = self.prev_status(row['ref'], self.max_times[0])
            if True:  # prev_status == "successful":
                try:
                    suitable = self.input_output_suitable_for_tabpfn(
                        schema, suggested_columns, data, row
                    )
                    if suitable is None:
                        continue
                    table, input_columns, output_column = suitable
                    six_tuple_dataset = self.convert_dataset_to_six_tuple(
                        schema, input_columns, output_column, table, row
                    )
                    try:
                        res = self.tabpfn_evaluate(six_tuple_dataset, False, True)
                        if len(global_results) == 0:
                            global_results.update(res)
                        else:
                            for k in res:
                                global_results[k].update(res[k])
                    except TabPFNError:
                        continue
                    dataset_names.append(six_tuple_dataset[0])
                    self.eval_ends(row['ref'], self.max_times[0])
                except:
                    continue
        with open(f"./global_results_{self.max_time}.pkl", 'wb') as f:
            pickle.dump((global_results, dataset_names), f)
            print("****** DUMPED PICKLE GLOBAL RESULTS *****")

    def fast_load_analyze(self):
        with open(f"./global_results_{self.max_time}.pkl", 'rb') as f:
            global_results, dataset_names = pickle.load(f)
        df = self.load_models_analyze_results(global_results, dataset_names)
        print(df)

    def identifier(self):
        pass


if __name__ == "__main__":
    ka = RunKaggle()
    ka.run_all_evaluation(overwrite=False)  # to run
    ka.analyze_and_save_results()
    ka.fast_load_analyze()
