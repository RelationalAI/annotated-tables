import pandas as pd

pd.DataFrame.iteritems = pd.DataFrame.items

from scripts import tabular_baselines

from ddatasets import load_openml_list, valid_dids_classification, test_dids_classification, open_cc_dids
from scripts.tabular_baselines import *
from scripts.tabular_evaluation import evaluate
from scripts import tabular_metrics
import warnings

from tabpfn.scripts.tabular_baselines_deep import saint_metric

warnings.filterwarnings("ignore")

from notebook_utils import *


def get_datasets(selector, task_type, suite='openml'):
    cc_test_datasets_multiclass, cc_test_datasets_multiclass_df = load_openml_list(open_cc_dids, multiclass=True,
                                                                                   shuffled=True, filter_for_nan=False,
                                                                                   max_samples=10000, num_feats=100,
                                                                                   return_capped=True)
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


def eval_method(datasets, task_type, method, eval_positions, max_time, metric_used, split_number,
                append_metric=True, fetch_only=False, verbose=False, overwrite=True, load_only=False,
                save=True):
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

    base_path = os.path.join('.')
    bptt = 2000
    device = 'cpu'

    clf = clf_dict[method]

    time_string = '_time_' + str(max_time) if max_time else ''
    metric_used_string = '_' + tabular_baselines.get_scoring_string(metric_used, usage='') if append_metric else ''

    result = evaluate(datasets=datasets
                      , model=clf
                      , method=method + time_string + metric_used_string
                      , bptt=bptt, base_path=base_path
                      , eval_positions=eval_positions
                      , device=device, max_splits=1
                      , overwrite=overwrite
                      , save=save
                      , metric_used=metric_used
                      , path_interfix=task_type
                      , fetch_only=fetch_only
                      , split_number=split_number
                      , verbose=verbose
                      , max_time=max_time,
                      load_only=load_only)

    return result


def load_all_kaggle_datasets(indices=None):
    adapter = KaggleAdapterTabPFN()
    selected_datasets = adapter.output_datasets()
    if indices is not None:
        selected = [selected_datasets[i] for i in indices]
    else:
        selected = selected_datasets
    return selected


def main():
    eval_positions = [1000]
    max_features = 100
    selector = 'test'
    overwrite = True
    max_times = [60, ]
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
    split_numbers = [1]

    if True:
        suite = 'cc'
        # RUN ALL METHODS, SPLITS AND DATASETS
        selected_datasets = get_datasets('test', task_type, suite=suite)
        selected_datasets = selected_datasets[1:2]
    else:
        selected_datasets = load_all_kaggle_datasets()

    jobs = []
    for m in methods:
        for max_time in max_times:
            for split_number in split_numbers:
                jobs.append(
                    eval_method(selected_datasets, task_type, m, eval_positions, max_time, metric_used,
                                split_number, overwrite=overwrite)
                )


if __name__ == '__main__':
    main()

# TODO:
#  load kaggle dataset
#  filter and adjust prompt to satisfy TabPFN requirements
#  make sure the results are saved
#  run on ten to measure the speed and stability
#  run on one thousand

"""
dataset: 6-tuple
name,
input feature tensor, float: row, feature. 2000 rows. 76 features
target tensor integer: row
empty list?
name of the input features 
dict: {'classes_capped': False, 'feats_capped': False, 'samples_capped': False}
"""
