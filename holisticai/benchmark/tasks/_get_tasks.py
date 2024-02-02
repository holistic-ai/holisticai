from holisticai.benchmark.utils import (
    BinaryClassificationBenchmark,
    ClusteringBenchmark,
    MultiClassificationBenchmark,
    RecommenderSystemBenchmark,
    RegressionBenchmark,
)

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
