import os
import sys

import pandas as pd

# Add the current directory to the sys path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def load_benchmark(task=None, type=None):
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
    filepath = os.path.join(current_dir, f"{type}_{task}_benchmark.parquet")
    return pd.read_parquet(filepath)
