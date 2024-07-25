from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error


class DataMinimizationAccuracyRatio:
    reference: float = np.inf
    name: str = "Data Minimization Accuracy Ratio"

    def __call__(self, y_true, y_pred, y_pred_dm, return_results=False):
        metrics_results = pd.DataFrame(
            [
                {
                    "Selection Type": yp["selector_type"],
                    "Modifier Type": yp["modifier_type"],
                    "N_feats": yp["n_feats"],
                    "Feats": yp["feats"],
                    "Score": relative_performance(accuracy_score, y_pred, yp["predictions"], y_true=y_true),
                    "Accuracy": accuracy_score(y_true, yp["predictions"]),
                }
                for yp in y_pred_dm
            ]
        )
        index = metrics_results["Score"].argmin()
        if return_results:
            pred_row = pd.DataFrame(
                [
                    {
                        "Selection Type": "Base",
                        "Modifier Type": "Base",
                        "N_feats": 0,
                        "Feats": [],
                        "Score": 1,
                        "Accuracy": accuracy_score(y_true, y_pred),
                    }
                ]
            )
            metrics_results = pd.concat([metrics_results, pred_row], ignore_index=True)
            return metrics_results, metrics_results["Score"].loc[index]
        return float(metrics_results["Score"].loc[index])


class DataMinimizationMSERatio:
    reference: float = 0
    name: str = "Data Minimization MSE Ratio"

    def __call__(self, y_true, y_pred, y_pred_dm, return_results=False):
        metrics_results = pd.DataFrame(
            [
                {
                    "Selection Type": yp["selector_type"],
                    "Modifier Type": yp["modifier_type"],
                    "N_feats": yp["n_feats"],
                    "Feats": yp["feats"],
                    "Score": relative_performance(mean_squared_error, y_pred, yp["predictions"], y_true=y_true),
                    "MSE": mean_squared_error(y_true, yp["predictions"]),
                }
                for yp in y_pred_dm
            ]
        )
        index = metrics_results["Score"].argmin()
        if return_results:
            pred_row = pd.DataFrame(
                [
                    {
                        "Selection Type": "Base",
                        "Modifier Type": "Base",
                        "N_feats": 0,
                        "Feats": [],
                        "Score": 1,
                        "MSE": mean_squared_error(y_true, y_pred),
                    }
                ]
            )
            metrics_results = pd.concat([metrics_results, pred_row], ignore_index=True)
            return metrics_results, metrics_results["Score"].loc[index]
        return float(metrics_results["Score"].loc[index])


def get_learning_task(y_true: pd.Series):
    if y_true.dtype.kind in ["i", "u", "O"]:
        return "classification"
    if y_true.dtype.kind in ["f"]:
        return "regression"
    raise ValueError(f"Unknown learning task. dtype: {y_true.dtype.kind}")


def data_minimization_score(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_pred_dm: dict[str, pd.Series],
    return_results=False,
    learning_task: Union[str, None] = None,
):
    """
    Calculate the accuracy ratio for data minimization. The accuracy ratio is the ratio of the accuracy of the data minimization model to the accuracy of the original model.

    Parameters
    ----------
    y_true: pd.Series
        The true labels.
    y_pred: pd.Series
        The predicted labels.
    y_pred_dm: dict[str, pd.Series]
        The predicted labels for each data minimization technique.
    return_results: bool
        Whether to return the results or not. Default is False.
    learning_task: str (Optional)
        The learning task. Can be either "classification" or "regression". If None, it will be inferred from the data.
    Returns
    -------
        float: The accuracy ratio for data minimization.
        pd.DataFrame: The results of the data minimization if return_results is True.
    """
    if learning_task is None:
        learning_task = get_learning_task(y_true)

    if learning_task == "classification":
        dm = DataMinimizationAccuracyRatio()
    if learning_task == "regression":
        dm = DataMinimizationMSERatio()
    return dm(y_true, y_pred, y_pred_dm, return_results)


def relative_performance(metric_fn, y_pred, y_pred_dm, y_true):
    """
    Parameters
    ----------
    metric_fn: function
        metric function used to compare.

    y_true: array-like
        vector-target

    y_pred_dm: array-like
        predicted vector fitted with data minimization

    y_pred: array-like
        predicted vector fitted with all features

    Return
    ------
        relative performance metric
    """
    y = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_pred_dm = np.array(y_pred_dm).flatten()
    return metric_fn(y, y_pred) / metric_fn(y, y_pred_dm)


def relative_clustering_performance(metric_fn, y_pred, y_pred_dm, x):
    """
    Parameters
    ----------
    metric_fn: function
        metric function used to compare.

    y_true: array-like
        vector-target

    y_pred_dm: array-like
        predicted vector fitted with data minimization

    X: array-like
        input matrix

    Return
    ------
        relative performance metric
    """
    y_pred = np.array(y_pred).flatten()
    y_pred_dm = np.array(y_pred_dm).flatten()
    try:
        dn = metric_fn(x, y_pred) / metric_fn(x, y_pred_dm)
    except:  # noqa: E722
        dn = np.nan

    return dn
