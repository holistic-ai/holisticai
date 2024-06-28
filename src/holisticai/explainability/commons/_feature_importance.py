from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
from holisticai.explainability.commons._definitions import (
    ConditionalFeatureImportance,
    FeatureImportance,
    Importances,
    LocalConditionalFeatureImportance,
    LocalImportances,
)

if TYPE_CHECKING:
    import pandas as pd
    from holisticai.datasets import Dataset


def filter_feature_importance(feature_importance, alpha=None) -> pd.DataFrame:
    """
    Filters the feature importance dataframe based on a given threshold.

    Args:
        feature_importance (pd.DataFrame): The feature importance dataframe.
        alpha (float, optional): The threshold value. Defaults to None.

    Returns:
        pd.DataFrame: The filtered feature importance dataframe.
    """
    if alpha is None:
        alpha = 0.8

    feature_importance = feature_importance.sort_values("Importance", ascending=False)

    importance = feature_importance["Importance"].abs()

    feature_weight = importance / sum(importance)

    accum_feature_weight = feature_weight.cumsum()

    threshold = max(accum_feature_weight.min(), alpha)

    return feature_importance.loc[accum_feature_weight <= threshold]


def compute_conditional_feature_importance(
    learning_task, ds: Dataset, feature_importance_calculator: callable
) -> Union[ConditionalFeatureImportance,LocalConditionalFeatureImportance]:
    ds = create_output_groups(ds, learning_task)

    if feature_importance_calculator.importance_type == "global":
        conditional_feature_importance = {group_name[0]: feature_importance_calculator(ds=group_ds)
        for group_name, group_ds in ds.groupby("group")
        }
        return ConditionalFeatureImportance(conditional_feature_importance=conditional_feature_importance)
    conditional_feature_importance = {
    group_name[0]: LocalImportances(feature_importances=feature_importance_calculator(ds=group_ds))
    for group_name, group_ds in ds.groupby("group")
    }
    return LocalConditionalFeatureImportance(conditional_feature_importance=conditional_feature_importance)


def compute_ranked_feature_importance(feature_importances: FeatureImportance) -> FeatureImportance:
    ranked_feature_importances = filter_feature_importance(feature_importances.feature_importances, alpha=0.8)
    ranked_feature_names = ranked_feature_importances.Variable.tolist()
    return Importances(feature_importances=ranked_feature_importances, feature_names=ranked_feature_names)


def quantil_classify(q1, q2, q3, labels, x):
    if x <= q1:
        return labels[0]
    if q1 < x <= q2:
        return labels[1]
    if q2 < x <= q3:
        return labels[2]
    return labels[3]


def create_output_groups(ds, learning_task="regression"):
    if learning_task in ["binary_classification", "multi_classification"]:
        return ds.map(lambda sample: {"group": str(sample["y"])}, vectorized=False)

    if learning_task in ["regression"]:
        labels = ["Q0-Q1", "Q1-Q2", "Q2-Q3", "Q3-Q4"]
        labels_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        v = np.array(ds["y"].quantile(labels_values)).squeeze()
        return ds.map(
            lambda sample: {"group": quantil_classify(v[1], v[2], v[3], labels, sample["y"])}, vectorized=False
        )
    raise ValueError(f"Learning task {learning_task} not supported")
