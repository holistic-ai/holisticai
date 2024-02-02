from ._binary_classification import BinaryClassificationBenchmark
from ._clustering import ClusteringBenchmark
from ._multiclass import MultiClassificationBenchmark
from ._recommender import RecommenderSystemBenchmark
from ._regression import RegressionBenchmark

__all__ = [
    "BinaryClassificationBenchmark",
    "MultiClassificationBenchmark",
    "RegressionBenchmark",
    "ClusteringBenchmark",
    "RecommenderSystemBenchmark",
]
