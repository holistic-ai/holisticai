from __future__ import annotations

from typing import Literal, Union

from holisticai.datasets import Dataset
from holisticai.explainability.commons import (
    RegressionClassificationXAISettings,
    compute_explainability_features,
)
from holisticai.explainability.metrics._utils import compute_explainability_metrics_from_features


def regression_explainability_features(
    X, y, predict_fn, strategy: Union[Literal["permutation", "surrogate"], callable] = "permutation"
):
    dataset = Dataset(X=X, y=y)

    learning_task_settings = RegressionClassificationXAISettings(predict_fn=predict_fn)

    return compute_explainability_features(
        dataset=dataset, learning_task_settings=learning_task_settings, strategy=strategy
    )


def regression_explainability_metrics(X, y, predict_fn, strategy: Literal["permutation", "surrogate"] = "permutation"):
    xai_features = regression_explainability_features(X, y, predict_fn, strategy=strategy)
    return compute_explainability_metrics_from_features(xai_features)
