import numpy as np
import pandas as pd


def quantil_classify(q1, q2, q3, labels, x):
    if x <= q1:
        return labels[0]
    if q1 < x <= q2:
        return labels[1]
    if q2 < x <= q3:
        return labels[2]
    return labels[3]


def group_samples_by_learning_task(y: pd.Series, learning_task="regression", return_group_mask=False):
    if learning_task in ["binary_classification", "multi_classification"]:
        y_group = y.astype(str)

    elif learning_task in ["regression"]:
        labels = ["Q0-Q1", "Q1-Q2", "Q2-Q3", "Q3-Q4"]
        labels_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        v = np.array(y.quantile(labels_values)).squeeze()
        y_group = y.map(lambda x: quantil_classify(v[1], v[2], v[3], labels, x))

    else:
        raise ValueError(f"Learning task {learning_task} not supported")

    if return_group_mask:
        return y_group
    return y_group.groupby(y_group).apply(lambda x: x.index.tolist()).to_dict()
