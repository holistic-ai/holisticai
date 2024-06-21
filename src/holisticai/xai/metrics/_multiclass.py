from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from holisticai.datasets import Dataset
from holisticai.xai.commons import (
    MultiClassificationXAISettings,
    compute_xai_features,
    select_feature_importance_strategy,
)
from holisticai.xai.metrics._utils import compute_xai_metrics_from_features

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def multiclass_xai_features(X, y, predict_fn, predict_proba_fn, classes, strategy : Literal["permutation","surrogate"]="permutation"):
    dataset = Dataset(X=X, y=y)
    learning_task_settings = MultiClassificationXAISettings(predict_fn=predict_fn, predict_proba_fn=predict_proba_fn, classes=classes)
    feature_importance_fn = select_feature_importance_strategy(learning_task_settings=learning_task_settings, strategy=strategy)
    return compute_xai_features(dataset=dataset, learning_task_settings=learning_task_settings, feature_importance_fn=feature_importance_fn)


def multiclass_xai_metrics(X:NDArray, y: ArrayLike, predict_fn: callable, predict_proba_fn: callable, classes: list|None=None, strategy : Literal["permutation","surrogate"]="permutation"):
    if classes is None:
        classes = list(range(len(set(y))))
    xai_features = multiclass_xai_features(X, y, predict_fn, predict_proba_fn, classes, strategy=strategy)
    return compute_xai_metrics_from_features(xai_features)
