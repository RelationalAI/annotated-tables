from tqdm import tqdm

from tabpfn.aproto import ChatGPTSuggestTargetColumn
from tabpfn.run_kaggle import RunKaggle

def count_attrition():
    rk = RunKaggle()

    columns = rk.proto.suggest_input_target(load=True)
    print(len(columns))

    total_suitable = 0
    for tupl in tqdm(rk.proto.kaggle_iterator(), total=len(rk.proto.selected_indices)):
        if tupl is None:
            continue
        schema, suggested_columns, data, row = tupl
        try:
            suitable = rk.input_output_suitable_for_tabpfn(
                schema, suggested_columns, data, row
            )
            if suitable is not None:
                total_suitable += 1
        except:
            pass
    print(total_suitable)


if __name__ == '__main__':
    count_attrition()
