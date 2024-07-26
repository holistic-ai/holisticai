import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def rmse_score(y_true, y_pred, **kargs):
    return np.sqrt(mean_squared_error(y_true, y_pred, **kargs))


def smape(y_true, y_pred):
    return 1.0 / len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def regression_efficacy_metrics(y_pred, y_true):
    """
    Regression efficacy metrics batch computation.

    Description
    -----------
    This function computes all the relevant regression efficacy metrics,
    and displays them as a pandas dataframe. It also includes a reference value for comparison.

    Parameters
    ----------
    y_pred : array-like
        Predictions vector
    y_true (optional) : array-like
        Target vector

    Returns
    -------
    pandas DataFrame
        Metrics | Values | Reference
    """
    from sklearn import metrics

    perform = {
        # "Pseudo-R2": pr2_score,
        "RMSE": rmse_score,
        "MAE": metrics.mean_absolute_error,
        "MAPE": metrics.mean_absolute_percentage_error,
        "Max Error": metrics.max_error,
        "SMAPE": smape,
    }

    hyper = {
        # "Pseudo-R2": {},
        "RMSE": {"sample_weight": None, "multioutput": "uniform_average"},
        "MAE": {"sample_weight": None, "multioutput": "uniform_average"},
        "MAPE": {"sample_weight": None, "multioutput": "uniform_average"},
        "Max Error": {},
        "SMAPE": {},
    }

    ref_vals = {
        "RMSE": 0,
        "MAE": 0,
        "MAPE": 0,
        "Max Error": 0,
        "SMAPE": 0,
    }

    metrics = [[pf, fn(y_pred, y_true, **hyper[pf]), ref_vals[pf]] for pf, fn in perform.items()]

    return pd.DataFrame(metrics, columns=["Metric", "Value", "Reference"]).set_index("Metric")
