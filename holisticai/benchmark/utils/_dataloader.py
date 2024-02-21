import os

import pandas as pd


def load_benchmark(task=None, type=None, ranking=False):
    valid_tasks = [
        "binary_classification",
        "multi_classification",
        "regression",
        "clustering",
        "recommender",
    ]
    valid_types = ["preprocessing", "inprocessing", "postprocessing"]
    if task not in valid_tasks:
        raise ValueError(
            f"Task {task} not found. Please provide a valid type: {', '.join(valid_tasks)}"
        )
    if type not in valid_types:
        raise ValueError(
            f"Type {type} not found. Please provide a valid type: {', '.join(valid_types)}"
        )
    benchmarkdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = f"{benchmarkdir}/baselines/{type}_{task}_benchmark.parquet"
    if ranking:
        data = pd.read_parquet(filepath)
        rank = abs(
            data.pivot_table(
                index="Mitigator",
                columns="Dataset",
                values="AFS",
                aggfunc="mean",
            )
        )
        rank.insert(0, "Average AFS", rank.mean(axis=1))
        return rank.sort_values(by="Average AFS", ascending=False)
    else:
        return pd.read_parquet(filepath)
