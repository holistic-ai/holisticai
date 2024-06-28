from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

from holisticai.datasets import Dataset
from holisticai.explainability.commons import (
    MultiClassificationXAISettings,
    compute_explainability_features,
)
from holisticai.explainability.metrics._utils import compute_explainability_metrics_from_features

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def multiclass_explainability_features(
    X, y, predict_fn, predict_proba_fn, classes, strategy: Union[Literal["permutation", "surrogate"],callable] = "permutation"
):
    dataset = Dataset(X=X, y=y)
    learning_task_settings = MultiClassificationXAISettings(
        predict_fn=predict_fn, predict_proba_fn=predict_proba_fn, classes=classes
    )

    return compute_explainability_features(
        dataset=dataset, learning_task_settings=learning_task_settings, strategy=strategy
    )


def multiclass_explainability_metrics(
    X: NDArray,
    y: ArrayLike,
    predict_fn: callable,
    predict_proba_fn: callable,
    classes: list | None = None,
    strategy: Literal["permutation", "surrogate"] = "permutation",
):
    if classes is None:
        classes = list(range(len(set(y))))
    xai_features = multiclass_explainability_features(X, y, predict_fn, predict_proba_fn, classes, strategy=strategy)
    return compute_explainability_metrics_from_features(xai_features)
