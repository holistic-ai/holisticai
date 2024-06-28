from __future__ import annotations

from typing import Literal, Union

from holisticai.datasets import Dataset
from holisticai.explainability.commons import (
    BinaryClassificationXAISettings,
    compute_explainability_features,
)
from holisticai.explainability.metrics._utils import (
    compute_explainability_metrics_from_features,
)


def classification_explainability_features(
    X,
    y,
    predict_fn,
    predict_proba_fn,
    classes,
    strategy: Union[Literal["permutation", "surrogate"], callable] = "permutation",
):
    dataset = Dataset(X=X, y=y)

    learning_task_settings = BinaryClassificationXAISettings(
        predict_fn=predict_fn, predict_proba_fn=predict_proba_fn, classes=classes
    )

    return compute_explainability_features(dataset, learning_task_settings=learning_task_settings, strategy=strategy)


def classification_explainability_metrics(
    X,
    y,
    predict_fn,
    predict_proba_fn,
    classes: list | None = None,
    strategy: Literal["permutation", "surrogate"] = "permutation",
):
    if classes is None:
        classes = [0, 1]

    xai_features = classification_explainability_features(
        X, y, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn, classes=classes, strategy=strategy
    )
    return compute_explainability_metrics_from_features(xai_features)
