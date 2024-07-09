from dask.distributed import Client
import pandas as pd
from scipy.stats import rankdata

from tabpfn.analyze_results import Correlation
from tabpfn.run_kaggle import RunKaggle

pd.DataFrame.iteritems = pd.DataFrame.items

from scripts import tabular_baselines

from ddatasets import load_openml_list, valid_dids_classification, test_dids_classification, open_cc_dids
from scripts.tabular_baselines import *
from scripts.tabular_evaluation import evaluate
from scripts.tabular_metrics import calculate_score, make_metric_matrix
from scripts import tabular_metrics

import warnings

from tabpfn.scripts.tabular_baselines_deep import saint_metric

warnings.filterwarnings("ignore")

from notebook_utils import *

cat18 = True
cc_test_datasets_multiclass, cc_test_datasets_multiclass_df = load_openml_list(open_cc_dids, multiclass=True,
                                                                               shuffled=True, filter_for_nan=cat18,
                                                                               max_samples=10000, num_feats=100,
                                                                               return_capped=True)


# 6 hours per dataset
# 1000 dataset = 250 days.


def get_datasets(selector, task_type, suite='openml'):
    if task_type == 'binary':
        ds = valid_datasets_binary if selector == 'valid' else test_datasets_binary
    else:
        if suite == 'openml':
            ds = valid_datasets_multiclass if selector == 'valid' else test_datasets_multiclass
        elif suite == 'cc':
            ds = valid_datasets_multiclass if selector == 'valid' else cc_test_datasets_multiclass
        else:
            raise Exception("Unknown suite")
    return ds


# # Setting params

# 60 seconds per model per dataset * 6 models * 10000 datasets  = 41 days

# autogluon and tabpfn with parallelization = 10 days


eval_positions = [1000]
max_features = 100
bptt = 2000
selector = 'test'
base_path = os.path.join('.')
overwrite = False
max_time_ = 300
max_times = [max_time_, ]
# [0.5, 1, 15, 30, 60, 60 * 5, 60 * 15, 60 * 60] # SCIENCE altered to reduce total time
metric_used = tabular_metrics.auc_metric
methods = [
    # 'transformer',# 'autosklearn2',
    #        'autogluon', 'lightgbm',
    #        'xgb',
    # 'logistic', 'cocktail',
    # 'catboost',
    # 'knn',
    # 'saint_metric',
    'transformer', 'autogluon'
]
# tabpfn, logreg,
# ['transformer', 'logistic', 'gp', 'knn', 'catboost', 'xgb', 'autosklearn2',
#            'autogluon']
# SCIENCE altered to match paper
task_type = 'multiclass'
split_numbers = [1, 2, 3, 4, 5]

suite = 'cc'
test_datasets = get_datasets('test', task_type, suite=suite)

# test_datasets = test_datasets[0:1]

if cat18:
    test_datasets_ = test_datasets
    test_datasets = [t for t in test_datasets if len(t[3]) == 0]

clf_dict = {'gp': gp_metric
    , 'knn': knn_metric
    , 'catboost': catboost_metric
    , 'xgb': xgb_metric
    , 'transformer': transformer_metric
    , 'logistic': logistic_metric
    , 'autosklearn': autosklearn_metric
    , 'autosklearn2': autosklearn2_metric
    , 'autogluon': autogluon_metric,
            'saint_metric': saint_metric,
            'cocktail': well_tuned_simple_nets_metric,
            'lightgbm': lightgbm_metric
            }

device = 'cuda'


def eval_method(task_type, method, ds, selector, eval_positions, max_time, metric_used, split_number,
                append_metric=True, fetch_only=False, verbose=False):
    # dids = dids if type(dids) is list else [dids]
    # 
    # for did in dids:
    #     ds = get_datasets(selector, task_type, suite=suite)
    # 
    #     ds = ds if did is None else ds[did:did + 1]
    ds = [ds]

    clf = clf_dict[method]

    time_string = '_time_' + str(max_time) if max_time else ''
    metric_used_string = '_' + tabular_baselines.get_scoring_string(metric_used, usage='') if append_metric else ''

    result = evaluate(datasets=ds
                      , model=clf
                      , method=method + time_string + metric_used_string
                      , bptt=bptt, base_path=base_path
                      , eval_positions=eval_positions
                      , device=device, max_splits=1
                      , overwrite=overwrite
                      , save=True
                      , metric_used=metric_used
                      , path_interfix=task_type
                      , fetch_only=fetch_only
                      , split_number=split_number
                      , verbose=verbose
                      , max_time=max_time)

    return result


# RUN ALL METHODS, SPLITS AND DATASETS
# test_datasets = get_datasets('test', task_type, suite=suite)
# test_datasets = test_datasets[1:2]

jobs = [
    eval_method(task_type, m, ds, selector, eval_positions, max_time, metric_used, split_number)
    for ds in test_datasets
    for selector in ['test']
    for m in methods
    for max_time in max_times
    for split_number in split_numbers  # , 2, 3, 4, 5]
]

pos = str(eval_positions[0])

global_results = {}
for ds in test_datasets:
    for method in methods:
        for max_time in max_times:
            for split_number in split_numbers:
                time_st = method + '_time_' + str(max_time) + tabular_baselines.get_scoring_string(metric_used,
                                                                                                   usage='') + '_split_' + str(
                    split_number)
                res = eval_method(task_type, method, ds, selector,
                                  eval_positions, fetch_only=True,
                                  verbose=False, max_time=max_time,
                                  metric_used=metric_used,
                                  split_number=split_number)
                if time_st not in global_results:
                    global_results[time_st] = res
                else:
                    global_results[time_st].update(res)
global_results_backup = global_results.copy()
# TODO this is not saved anywhere else in the project,
#   Need to turn global_results into this pkl file?
# TODO need to set the identifier correctly, instead of numbers


# Verify integrity of results
# for bl in set(global_results.keys()):
#     if 'split_1' in bl:
#         for ds in test_datasets:
#             if f'{ds[0]}_ys_at_1000' not in global_results[bl]:
#                 continue
#             match = (global_results[bl][f'{ds[0]}_ys_at_1000'] == global_results['transformer_split_1'][
#                 f'{ds[0]}_ys_at_1000']).float().mean()
#             if not match:
#                 raise Exception("Not the same labels used")

limit_to = ''
test_datasets_names = [test_dataset[0] for test_dataset in test_datasets]
calculate_score(tabular_metrics.auc_metric, 'roc', global_results, test_datasets, eval_positions + [-1],
                limit_to=limit_to)
calculate_score(tabular_metrics.cross_entropy, 'cross_entropy', global_results, test_datasets, eval_positions + [-1],
                limit_to=limit_to)
calculate_score(tabular_metrics.accuracy_metric, 'acc', global_results, test_datasets, eval_positions + [-1])
calculate_score(tabular_metrics.time_metric, 'time', global_results, test_datasets, eval_positions + [-1],
                aggregator='sum', limit_to=limit_to)
calculate_score(tabular_metrics.time_metric, 'time', global_results, test_datasets, eval_positions + [-1],
                aggregator='mean', limit_to=limit_to)
calculate_score(tabular_metrics.count_metric, 'count', global_results, test_datasets, eval_positions + [-1],
                aggregator='sum', limit_to=limit_to)


# #### ROC and AUC plots from TabPFN Paper
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


def generate_ranks_and_wins_table(global_results_filtered, metric_key, max_time, split_number, time_matrix):
    global_results_filtered_split = {**global_results_filtered}
    time_str = '_time_' + str(max_time) + tabular_baselines.get_scoring_string(
        metric_used,
        usage='') + '_split_' + str(
        split_number)
    global_results_filtered_split = {k: global_results_filtered_split[k] for k in
                                     global_results_filtered_split.keys()
                                     if time_str in k or 'transformer_split_' + str(split_number) in k}

    matrix, matrix_stds, _ = make_metric_matrix(global_results_filtered_split, methods, pos, metric_key, test_datasets)
    for method in methods:
        if time_matrix[method] > max_time * 2:
            matrix[method] = np.nan
        # = np.nan
    # TODO where does na come from?

    if metric_key == 'cross_entropy':
        matrix = (matrix.fillna(100))
        higher_the_better = False
    elif metric_key == 'time':
        higher_the_better = False
    else:
        matrix = matrix.fillna(-1)
        higher_the_better = True
    # TODO matrix is 18 rows with all rows
    # why is it missing rows?
    ranks, wins = make_ranks_and_wins_table(matrix.copy(), higher_the_better=higher_the_better)
    return ranks, wins


df_ = []
metric_keys = ['roc', 'cross_entropy', 'time']

for max_time in max_times:
    global_results_filtered = {**global_results}
    global_results_filtered = {k: global_results_filtered[k] for k in global_results_filtered.keys() if
                               '_time_' + str(max_time) + tabular_baselines.get_scoring_string(metric_used,
                                                                                               usage='') + '_' in k or 'transformer' in k}

    time_matrix, _, _ = make_metric_matrix(global_results_filtered, methods, pos, 'time', test_datasets)
    time_matrix = time_matrix.mean()

    if len(global_results_filtered) == 0:
        continue

    # Calculate ranks and wins per split
    for metric_key in metric_keys:
        for split_number in split_numbers:
            ranks, wins = generate_ranks_and_wins_table(global_results_filtered, metric_key, max_time, split_number,
                                                        time_matrix)

            for method in methods:
                method_ = method + '_time_' + str(max_time) + tabular_baselines.get_scoring_string(metric_used,
                                                                                                   usage='')
                global_results[method_ + '_split_' + str(split_number)]['mean_rank_' + metric_key + f'_at_{pos}'] = \
                    ranks[method]
                global_results[method_ + '_split_' + str(split_number)]['mean_wins_' + metric_key + f'_at_{pos}'] = \
                    wins[method]

    # for method in global_results.keys():
    #    global_results[method]['mean_rank_'+metric_key+f'_at_{pos}'] = ranks[]

    avg_times = {}
    for method_ in methods:
        avg_times[method_] = []
        for split_number in split_numbers:
            method = method_ + '_time_' + str(max_time) + tabular_baselines.get_scoring_string(metric_used,
                                                                                               usage='') + '_split_' + str(
                split_number)
            avg_times[method_] += [global_results[method][f'mean_time_at_{pos}']]
    avg_times = pd.DataFrame(avg_times).mean()

    for metric_key in metric_keys:
        for ranking in ['', 'rank_', 'wins_']:
            for method_ in methods:
                for split_number in split_numbers:
                    method = method_
                    method = method_ + '_time_' + str(max_time) + tabular_baselines.get_scoring_string(metric_used,
                                                                                                       usage='') + '_split_' + str(
                        split_number)

                    if global_results[method][f'sum_count_at_{pos}'] <= 29:
                        print('Warning not all datasets generated for ' + method + ' ' + str(
                            global_results[method][f'sum_count_at_{pos}']))

                    time = global_results[method]['mean_time'] if ranking == '' else max_time
                    time = max_time  # Todo: This is not the real time
                    # df_ += [{'metric' + ranking + metric_key: global_results[method][
                    #     'mean_' + ranking + metric_key + f'_at_{pos}'], 'real_time': avg_times[method_], 'time': time,
                    #          'method': method_, 'split_number': split_number}]
                    #
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
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

rk = RunKaggle()
# rk.load_models_analyze_results(global_results, test_datasets_names)
# output = Correlation(max_time_).dump_results(global_results, test_datasets_names, "openml18", factors=False)

df_ = pd.DataFrame(df_)
print(df_)

# nans in roc cause some wins to not appear?
# investigate whether the nans can be reduced.
df_[(df_["method"] == "autogluon") & (df_["ranking"] == "default") & (df_["metric_key"] == "roc")][
    'metric_ranking_value'].mean()
