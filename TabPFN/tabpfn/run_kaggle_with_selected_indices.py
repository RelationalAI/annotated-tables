import sys

from run_kaggle import *


def run_indices(start, end):
    ka = RunKaggle(start, end)
    ka.run_all_evaluation(overwrite=False)


if __name__ == '__main__':
    start = sys.argv[1]
    end = sys.argv[2]
    start = int(start)
    end = int(end)
    run_indices(start, end)
