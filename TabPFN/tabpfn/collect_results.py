# 10 minutes tabpfn and baseline models have been run, and collect results here
from run_kaggle import RunKaggle


def save_and_print_tables():
    ka = RunKaggle()
    ka.main(overwrite=False, analyze=True)
    ka.fast_load_analyze()  # to analyze


if __name__ == '__main__':
    save_and_print_tables()
