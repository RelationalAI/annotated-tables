import json

import pandas
def a():
    path = "/mnt/d/Git/nirel/tabular/suggested_input_target_prototype7.json"
    # suggested_input_target_prototype7 = pandas.read_json(path, orient='records')
    # print(suggested_input_target_prototype7)
    with open(path, 'r') as f:
        data = json.loads(f.read())
    refactor = {'ref':[],
                "input_columns":[],
                "output_column":[],}
    for k, v in data.items():
        if 'input_columns' not in v or 'output_column' not in v:
            continue
        refactor['ref'].append(k)
        refactor['input_columns'].append(v['input_columns'])
        refactor['output_column'].append(v['output_column'])
    df = pandas.DataFrame( refactor)
    print(df)
    df.to_json('/mnt/d/Git/Annotated Tables/data/classification/input_output_columns.json')

if __name__ == '__main__':
    a()    