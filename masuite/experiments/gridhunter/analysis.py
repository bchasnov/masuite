import os
from numpy import result_type
import pandas as pd

def simple_score(file_path):
    file_data = pd.read_csv(file_path)
    count = 0
    for ret in file_data["agent0_avg_rets"]:
        if ret > 0:
            count += 1
    return round(count / len(file_data["agent0_avg_rets"]), 2)

def score_gridsearch(results_dir: str):
    results_df = pd.DataFrame(columns=["filename", "score"])
    for file in os.listdir(results_dir):
        filename = os.fsdecode(file)
        if not filename.endswith(".csv"):
            continue # raise value error?
        score = simple_score(f"{results_dir}/{filename}")
        results_df.loc[len(results_df.index)] = [filename, score]
    return results_df
