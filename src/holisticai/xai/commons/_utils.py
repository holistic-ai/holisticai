from collections import namedtuple
from typing import Literal

from holisticai.xai.commons import PermutationFeatureImportanceCalculator, SurrogateFeatureImportanceCalculator
from holisticai.xai.commons._definitions import LearningTaskXAISettings
from holisticai.xai.commons._feature_importance import (
    compute_conditional_feature_importance,
    compute_ranked_feature_importance,
)
from holisticai.xai.commons._partial_dependence import compute_partial_dependence
from numpy.random import RandomState


def select_feature_importance_strategy(
    learning_task_settings: LearningTaskXAISettings, strategy: Literal["permutation", "surrogate"]
) -> callable:
    if strategy == "permutation":
        return PermutationFeatureImportanceCalculator(
            learning_task_settings=learning_task_settings, random_state=RandomState(42)
        )

    if strategy == "surrogate":
        return SurrogateFeatureImportanceCalculator(
            learning_task_settings=learning_task_settings, random_state=RandomState(42)
        )
    raise ValueError(f"Invalid feature importance strategy: {strategy}")


def compute_xai_features(dataset, learning_task_settings: LearningTaskXAISettings, feature_importance_fn: callable):
    feature_importance = feature_importance_fn(ds=dataset)
    ranked_feature_importance = compute_ranked_feature_importance(feature_importance)
    conditional_feature_importance = compute_conditional_feature_importance(
        learning_task=learning_task_settings.learning_task, ds=dataset, feature_importance_fn=feature_importance_fn
    )
    partial_dependence = compute_partial_dependence(
        x=dataset["X"], features=feature_importance.feature_names, learning_task_settings=learning_task_settings
    )
    XAIFeatures = namedtuple(  # noqa: PYI024
        "XAIFeatures",
        ["feature_importance", "ranked_feature_importance", "conditional_feature_importance", "partial_dependence"],
    )
    return XAIFeatures(
        feature_importance=feature_importance,
        ranked_feature_importance=ranked_feature_importance,
        conditional_feature_importance=conditional_feature_importance,
        partial_dependence=partial_dependence,
    )
