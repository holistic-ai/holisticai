from __future__ import annotations

from typing import Literal

from holisticai.datasets import Dataset
from holisticai.xai.commons import (
    BinaryClassificationXAISettings,
    compute_xai_features,
    select_feature_importance_strategy,
)
from holisticai.xai.metrics._utils import compute_xai_metrics_from_features


def classification_xai_features(
    X, y, predict_fn, predict_proba_fn, classes, strategy: Literal["permutation", "surrogate"] = "permutation"
):
    dataset = Dataset(X=X, y=y)
    learning_task_settings = BinaryClassificationXAISettings(
        predict_fn=predict_fn, predict_proba_fn=predict_proba_fn, classes=classes
    )
    feature_importance_fn = select_feature_importance_strategy(
        learning_task_settings=learning_task_settings, strategy=strategy
    )
    return compute_xai_features(
        dataset, learning_task_settings=learning_task_settings, feature_importance_fn=feature_importance_fn
    )


def classification_xai_metrics(
    X,
    y,
    predict_fn,
    predict_proba_fn,
    classes: list | None = None,
    strategy: Literal["permutation", "surrogate"] = "permutation",
):
    if classes is None:
        classes = [0, 1]
    xai_features = classification_xai_features(
        X, y, predict_fn=predict_fn, predict_proba_fn=predict_proba_fn, classes=classes, strategy=strategy
    )
    return compute_xai_metrics_from_features(xai_features)
