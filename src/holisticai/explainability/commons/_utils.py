from __future__ import annotations

from collections import namedtuple
from typing import Literal, Union

from holisticai.explainability.commons import (
    PermutationFeatureImportanceCalculator,
    SurrogateFeatureImportanceCalculator,
)
from holisticai.explainability.commons._definitions import (
    LearningTaskXAISettings,
)
from holisticai.explainability.commons._feature_importance import (
    compute_global_conditional_feature_importance,
    compute_local_feature_importance,
    compute_ranked_feature_importance,
)
from holisticai.explainability.commons._partial_dependence import compute_partial_dependence
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


def global_feature_importances(
    dataset, learning_task_settings: LearningTaskXAISettings, feature_importance_calculator: callable
):
    feature_importance = feature_importance_calculator(ds=dataset)
    conditional_feature_importance = compute_global_conditional_feature_importance(
        learning_task=learning_task_settings.learning_task,
        ds=dataset,
        feature_importance_calculator=feature_importance_calculator,
    )
    return feature_importance, conditional_feature_importance


def compute_explainability_features(
    dataset,
    learning_task_settings: LearningTaskXAISettings,
    strategy: Union[Literal["permutation", "surrogate"], callable] = "permutation",
):
    if isinstance(strategy, str):
        feature_importance_calculator = select_feature_importance_strategy(
            learning_task_settings=learning_task_settings, strategy=strategy
        )
    elif callable(strategy):
        feature_importance_calculator = strategy
    else:
        raise TypeError(f"Invalid strategy {strategy}")

    if feature_importance_calculator.importance_type == "global":
        feature_importance, conditional_feature_importance = global_feature_importances(
            dataset=dataset,
            learning_task_settings=learning_task_settings,
            feature_importance_calculator=feature_importance_calculator,
        )
        local_feature_importance, local_conditional_feature_importance = None, None

    elif feature_importance_calculator.importance_type == "local":
        local_feature_importance, local_conditional_feature_importance = compute_local_feature_importance(
            dataset=dataset,
            learning_task=learning_task_settings.learning_task,
            feature_importance_calculator=feature_importance_calculator,
        )
        feature_importance = local_feature_importance.to_global()
        conditional_feature_importance = local_conditional_feature_importance.to_global()

    else:
        raise ValueError(f"Invalid feature importance type: {feature_importance_calculator.importance_type}")

    ranked_feature_importance = compute_ranked_feature_importance(feature_importance)

    partial_dependence = compute_partial_dependence(
        x=dataset["X"], features=ranked_feature_importance.feature_names, learning_task_settings=learning_task_settings
    )
    XAIFeatures = namedtuple(  # noqa: PYI024
        "XAIFeatures",
        [
            "feature_importance",
            "ranked_feature_importance",
            "conditional_feature_importance",
            "partial_dependence",
            "local_feature_importance",
            "local_conditional_feature_importance",
        ],
    )
    return XAIFeatures(
        feature_importance=feature_importance,
        ranked_feature_importance=ranked_feature_importance,
        conditional_feature_importance=conditional_feature_importance,
        partial_dependence=partial_dependence,
        local_feature_importance=local_feature_importance,
        local_conditional_feature_importance=local_conditional_feature_importance,
    )
