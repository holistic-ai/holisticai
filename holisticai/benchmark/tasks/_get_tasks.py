from ._binary_classification import BinaryClassificationBenchmark
from ._clustering import ClusteringBenchmark
from ._multiclass import MultiClassificationBenchmark
from ._recommender import RecommenderSystemBenchmark
from ._regression import RegressionBenchmark

TASKS = {
    "binary_classification": BinaryClassificationBenchmark(),
    "multiclass_classification": MultiClassificationBenchmark(),
    "regression": RegressionBenchmark(),
    "clustering": ClusteringBenchmark(),
    "recommender": RecommenderSystemBenchmark(),
}

task_name = list(TASKS.keys())


def get_task(task):
    valid_types = list(TASKS.keys())
    if task not in valid_types:
        raise ValueError(
            f"Task {task} not found. Please provide a valid type: {', '.join(valid_types)}"
        )
    return TASKS[task]
