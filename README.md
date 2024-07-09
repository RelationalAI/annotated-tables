# AnnotatedTables: A Large Tabular Dataset with Language Model Annotations

Paper: https://arxiv.org/abs/2406.16349


## Installation

conda create -n annotatedtables python=3.10
conda activate annotatedtables
pip install tqdm jsonlines pytest pytest-pycharm networkx prodict records babel tabulate openai
pip install ray openai rai-sdk datasets accelerate fuzzywuzzy python-Levenshtein kaggle wrapt-timeout-decorator
conda install pandas sqlalchemy

## Project structure
./sql contains the dataset construction code for SQL annotation and SQL-to-Rel translation
./llm contains the files needed to query ChatGPT and use the large language model
./rel contains files needed to run Rel code to evaluate SQL-to-Rel translation accuracy with execution accuracy
./worksheets contains scripts for reproducing the figures and tables in the paper

## How to run the project and build my own AnnotatedTables?
Go to ./sql/synth.py and the main method is SynthesisPipeline.run_from_step().
Depending on the pipeline step you choose, you may catalog the Kaggle datasets, download the Kaggle datasets,
get the schema and example row descriptions, synthesize SQL queries, filter SQL queries with execution,
few-shot translation of SQL to Rel, evaluate Rel execution accuracy, etc..

Note that the Kaggle datasets you catalog may be different from ours, as Kaggle is a platform with new datasets added 
everyday.

## Data
The data can be found on Zenodo: https://zenodo.org/records/11626802

