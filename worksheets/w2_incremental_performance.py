"""
Randomly select 1000 SQL programs.
For incremental in range(0, 45):
    prompt = None
    rel = KaggleSynth.translate_to_rel(programs, prompt)
    rel_translation_accuracy = accuracy(rel, sql)
"""
import json

import pandas

from sql.synth import KagglePrompting, SynthesisPipeline


def incremental_performance():
    prompt = KagglePrompting().manual_examples()
    pipeline = SynthesisPipeline(incremental_manual_examples=None)
    step = 5
    # 1
    pipeline.kaggle.download_datasets(step)
    # 2
    pipeline.get_all_schemas_and_top_rows(step)
    # 3
    pipeline.chatgpt_synthesize_sql_queries(step)
    # 4
    pipeline.filter_sql_execution_with_ground_truth(step)
    # selected_rows_and_total = pipeline.select_rows_and_total()
    # print(json.dumps([s[0] for s in selected_rows_and_total]))
    indices = [14592, 3278, 32098, 18289, 13434, 4165, 3905, 12280, 28657, 28893, 44597, 36421, 20379, 44118, 13396, 12156, 12676, 5695, 16361, 10328, 29871, 37930, 30512, 49823, 36434, 59429, 46566, 27460, 34993, 21417, 60589, 49735, 28785, 42504, 7331, 30021, 4207, 52581, 8675, 41245, 51856, 18726, 18301, 32325, 56155, 66784, 64686, 20969, 8326, 50019, 61348, 69352, 32953, 1504, 34973, 44587, 14621, 38469, 425, 66542, 39117, 74, 47576, 31385, 31571, 11226, 63699, 16483, 62296, 21643, 55461, 52296, 57422, 32493, 8392, 30161, 7716, 8834, 43309, 9287, 31195, 36500, 28080, 31850, 53883, 7100, 7944, 52772, 44473, 24931, 32742, 9880, 1934, 30982, 63092, 7685, 282, 51173, 20289, 38890, 7665, 7989, 41104, 62493, 66562, 10500, 24356, 52923, 32271, 54948, 41467, 41180, 31285, 51876, 17154, 9508, 1220, 60068, 9602, 27938, 66307, 45745, 9016, 48434, 37353, 20676, 39240, 13577, 34664, 20374, 35697, 36930, 27607, 44942, 26685, 34600, 64032, 12097, 36265, 43719, 17146, 14663, 9862, 19536, 5229, 27535, 53264, 20257, 31029, 21299, 54040, 32527, 34970, 50140, 29154, 26158, 45830, 29831, 29219, 3101, 36517, 9099, 66768, 15118, 56959, 41114, 67033, 50488, 33386, 221, 68146, 25825, 47739, 9171, 41145, 16334, 39363, 16683, 22810, 53225, 53, 39829, 37606, 27549, 62021, 22241, 12240, 30784, 19313, 59692, 25485, 64799, 13970, 23053, 32662, 59829, 17477, 69616, 33972, 31358]
    indices = [14592, 3278, 18289, 13434, 4165, 3905, 12280, 28657, 28893, 44597, 36421, 20379, 44118, 13396, 12156, 12676, 5695, 16361, 10328, 29871, 37930, 30512, 49823, 36434, 59429, 46566, 27460, 34993, 21417, 60589, 49735, 28785, 42504, 7331, 30021, 52581, 8675, 41245, 51856, 18726, 18301, 32325, 56155, 66784, 64686, 20969, 8326, 50019, 61348, 69352, 32953, 1504, 34973, 44587, 14621, 38469, 425, 66542, 39117, 74, 47576, 31385, 31571, 11226, 63699, 16483, 62296, 21643, 55461, 52296, 57422, 32493, 8392, 30161, 7716, 8834, 43309, 9287, 31195, 36500, 28080, 31850, 53883, 7100, 7944, 52772, 44473, 24931, 32742, 9880, 1934, 30982, 63092, 7685, 282, 51173, 20289, 38890, 7665, 7989, 41104, 62493, 66562, 10500, 24356, 52923, 32271, 54948, 41467, 41180, 31285, 51876, 17154, 9508, 1220, 60068, 9602, 27938, 66307, 45745, 9016, 48434, 37353, 20676, 39240, 13577, 34664, 20374, 35697, 36930, 27607, 44942, 26685, 34600, 64032, 12097, 36265, 43719, 17146, 14663, 9862, 19536, 5229, 27535, 53264, 20257, 31029, 21299, 54040, 32527, 34970, 50140, 29154, 26158, 45830, 29831, 29219, 3101, 36517, 9099, 66768, 15118, 56959, 41114, 67033, 50488, 33386, 221, 68146, 25825, 47739, 9171, 41145, 16334, 39363, 16683, 22810, 53225, 53, 39829, 37606, 27549, 62021, 22241, 12240, 30784, 19313, 59692, 25485, 64799, 13970, 23053, 32662, 59829, 17477, 69616, 33972, 31358]
    selected_rows_and_total = pipeline.manual_select_rows_and_total(indices)

    for manual_examples_step in range(30, len(prompt)+1, 5): # 10 runs
        pipeline = SynthesisPipeline(incremental_manual_examples=manual_examples_step)
        pipeline.kaggle.all_catalog_datasets()
        step = 5
        # 1
        pipeline.kaggle.download_datasets(step)
        # 2
        pipeline.get_all_schemas_and_top_rows(step)
        # 3
        pipeline.chatgpt_synthesize_sql_queries(step)
        # 4
        pipeline.filter_sql_execution_with_ground_truth(step)

        # TODO need to set up catalog and sql queries correctly to reflect the selected datasets.
        #   we select 1000 to evaluate: 0.4% of the workload last time
        #   4% in total.
        pipeline.chatgpt_translate_sql_to_rel(step, selected_indices_and_rows=selected_rows_and_total)
        pipeline.execute_rel_accuracy(step)
        print(" ========================="+str(manual_examples_step) + " step is done =========================")

# if __name__ == '__main__':
#     incremental_performance()



def performance_convergence_incremental_prompt_engineering():
    prompt = KagglePrompting().manual_examples()
    pipeline = SynthesisPipeline(incremental_manual_examples=None)
    step = 5
    # 1
    pipeline.kaggle.download_datasets(step)
    # 2
    pipeline.get_all_schemas_and_top_rows(step)
    # 3
    pipeline.chatgpt_synthesize_sql_queries(step)
    # 4
    pipeline.filter_sql_execution_with_ground_truth(step)
    # selected_rows_and_total = pipeline.select_rows_and_total()
    # print(json.dumps([s[0] for s in selected_rows_and_total]))
    indices = [14592, 3278, 32098, 18289, 13434, 4165, 3905, 12280, 28657, 28893, 44597, 36421, 20379, 44118, 13396,
               12156, 12676, 5695, 16361, 10328, 29871, 37930, 30512, 49823, 36434, 59429, 46566, 27460, 34993, 21417,
               60589, 49735, 28785, 42504, 7331, 30021, 4207, 52581, 8675, 41245, 51856, 18726, 18301, 32325, 56155,
               66784, 64686, 20969, 8326, 50019, 61348, 69352, 32953, 1504, 34973, 44587, 14621, 38469, 425, 66542,
               39117, 74, 47576, 31385, 31571, 11226, 63699, 16483, 62296, 21643, 55461, 52296, 57422, 32493, 8392,
               30161, 7716, 8834, 43309, 9287, 31195, 36500, 28080, 31850, 53883, 7100, 7944, 52772, 44473, 24931,
               32742, 9880, 1934, 30982, 63092, 7685, 282, 51173, 20289, 38890, 7665, 7989, 41104, 62493, 66562, 10500,
               24356, 52923, 32271, 54948, 41467, 41180, 31285, 51876, 17154, 9508, 1220, 60068, 9602, 27938, 66307,
               45745, 9016, 48434, 37353, 20676, 39240, 13577, 34664, 20374, 35697, 36930, 27607, 44942, 26685, 34600,
               64032, 12097, 36265, 43719, 17146, 14663, 9862, 19536, 5229, 27535, 53264, 20257, 31029, 21299, 54040,
               32527, 34970, 50140, 29154, 26158, 45830, 29831, 29219, 3101, 36517, 9099, 66768, 15118, 56959, 41114,
               67033, 50488, 33386, 221, 68146, 25825, 47739, 9171, 41145, 16334, 39363, 16683, 22810, 53225, 53, 39829,
               37606, 27549, 62021, 22241, 12240, 30784, 19313, 59692, 25485, 64799, 13970, 23053, 32662, 59829, 17477,
               69616, 33972, 31358]
    indices = [14592, 3278, 18289, 13434, 4165, 3905, 12280, 28657, 28893, 44597, 36421, 20379, 44118, 13396, 12156,
               12676, 5695, 16361, 10328, 29871, 37930, 30512, 49823, 36434, 59429, 46566, 27460, 34993, 21417, 60589,
               49735, 28785, 42504, 7331, 30021, 52581, 8675, 41245, 51856, 18726, 18301, 32325, 56155, 66784, 64686,
               20969, 8326, 50019, 61348, 69352, 32953, 1504, 34973, 44587, 14621, 38469, 425, 66542, 39117, 74, 47576,
               31385, 31571, 11226, 63699, 16483, 62296, 21643, 55461, 52296, 57422, 32493, 8392, 30161, 7716, 8834,
               43309, 9287, 31195, 36500, 28080, 31850, 53883, 7100, 7944, 52772, 44473, 24931, 32742, 9880, 1934,
               30982, 63092, 7685, 282, 51173, 20289, 38890, 7665, 7989, 41104, 62493, 66562, 10500, 24356, 52923,
               32271, 54948, 41467, 41180, 31285, 51876, 17154, 9508, 1220, 60068, 9602, 27938, 66307, 45745, 9016,
               48434, 37353, 20676, 39240, 13577, 34664, 20374, 35697, 36930, 27607, 44942, 26685, 34600, 64032, 12097,
               36265, 43719, 17146, 14663, 9862, 19536, 5229, 27535, 53264, 20257, 31029, 21299, 54040, 32527, 34970,
               50140, 29154, 26158, 45830, 29831, 29219, 3101, 36517, 9099, 66768, 15118, 56959, 41114, 67033, 50488,
               33386, 221, 68146, 25825, 47739, 9171, 41145, 16334, 39363, 16683, 22810, 53225, 53, 39829, 37606, 27549,
               62021, 22241, 12240, 30784, 19313, 59692, 25485, 64799, 13970, 23053, 32662, 59829, 17477, 69616, 33972,
               31358]
    selected_rows_and_total = pipeline.manual_select_rows_and_total(indices)
    accs = []
    for manual_examples_step in range(0, len(prompt) + 1, 5):  # 10 runs
        pipeline = SynthesisPipeline(incremental_manual_examples=manual_examples_step)
        pipeline.kaggle.all_catalog_datasets()
        step = 8
        # 1
        pipeline.kaggle.download_datasets(step)
        # 2
        pipeline.get_all_schemas_and_top_rows(step)
        # 3
        pipeline.chatgpt_synthesize_sql_queries(step)
        # 4
        pipeline.filter_sql_execution_with_ground_truth(step)

        pipeline.chatgpt_translate_sql_to_rel(step, selected_indices_and_rows=selected_rows_and_total)
        pipeline.execute_rel_accuracy(step)
        accs.append(pipeline.rel_execution_accuracy)

    df = {
        "Examples":[],
        "% Rel Correct":[],
        "% Rel Error": [],
        "% Diff. Results": [],
    }
    for step, acc in enumerate(accs):
        df["Examples"].append(step * 5)
        infos = acc[2]
        total = 2000
        correct =0
        diff = 0
        for info in infos:
            for i in info:
                if i == "correct":
                    correct +=1
                elif i == "comparison":
                    diff+=1
        error = total - correct - diff
        df["% Rel Correct"].append(100*correct/total)
        df["% Rel Error"].append(100*error/total)
        df["% Diff. Results"].append(100*diff/total)

    df = pandas.DataFrame(df)
    df.to_csv("incremental.csv", index=False)
    print("DONE")
    
if __name__ == '__main__':
    performance_convergence_incremental_prompt_engineering()