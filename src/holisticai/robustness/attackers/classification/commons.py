from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd


def x_array_to_df(x_arr, feature_names):
    return pd.DataFrame(x_arr, columns=feature_names)


def x_to_nd_array(x: pd.DataFrame):
    return np.array(x)


def to_categorical(labels: Union[np.ndarray, list[float]], nb_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.

    Parameters
    ----------
    labels : Union[np.ndarray, list[float]]
        An array of integer labels of shape `(nb_samples,)`.
    nb_classes : int, optional
        The number of classes (possible labels), by default None.

    Returns
    -------
    np.ndarray
        A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=int)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical


def format_function_predict_proba(learning_task, predict_proba_fn):
    """
    Format the predict_proba function based on the learning task.

    Parameters
    ----------
    learning_task : str
        The learning task.
    predict_proba_fn : callable
        The predict_proba function.

    Returns
    -------
    callable
        The formatted predict_proba function.
    """
    if learning_task == "binary_classification":

        def forward(x: np.ndarray, feature_names: list[str]):
            x_df = x_array_to_df(x, feature_names=feature_names)
            score = np.array(predict_proba_fn(x_df))
            if score.ndim == 2:
                return score
            return np.stack([1 - score, score], axis=1)

    return forward
