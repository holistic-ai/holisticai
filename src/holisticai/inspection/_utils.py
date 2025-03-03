from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def quantil_classify(q1, q2, q3, labels, x):
    if x <= q1:
        return labels[0]
    if q1 < x <= q2:
        return labels[1]
    if q2 < x <= q3:
        return labels[2]
    return labels[3]


def group_index_samples_by_learning_task(y: pd.Series, learning_task="regression") -> dict[str, list[int]]:
    y_group = group_mask_samples_by_learning_task(y, learning_task)
    unique_values = np.unique(y_group)
    return {val: np.where(y_group == val)[0].tolist() for val in unique_values}


def group_mask_samples_by_learning_task(
    y: pd.Series,
    learning_task="regression",
) -> pd.Series:
    if learning_task in ["binary_classification", "multi_classification"]:
        return y.astype(str)

    if learning_task in ["regression"]:
        labels = ["Q0-Q1", "Q1-Q2", "Q2-Q3", "Q3-Q4"]
        labels_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        v = np.array(y.quantile(labels_values)).squeeze()
        return y.map(lambda x: quantil_classify(v[1], v[2], v[3], labels, x))

    msg = f"Learning task {learning_task} not supported"
    raise ValueError(msg)
