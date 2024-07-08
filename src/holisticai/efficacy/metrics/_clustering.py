import numpy as np
import pandas as pd


def clustering_efficacy_metrics(X, y_pred):
    """
    Clustering efficacy metrics batch computation.

    Description
    -----------
    This function computes all the relevant clustering efficacy metrics,
    and displays them as a pandas dataframe. It also includes a reference value for comparison.

    Parameters
    ----------
    X : matrix-like
        feature matrix
    y_pred (optional) :  array-like
        Predicted vector

    Returns
    -------
    pandas DataFrame
        Metrics | Values | Reference

    Examples
    --------
    """
    from sklearn import metrics

    perform = {
        "Silhouette": metrics.silhouette_score,
        "Calinski Harabasz": metrics.calinski_harabasz_score,
        "Davies Bouldin": metrics.davies_bouldin_score,
    }

    hyper = {
        "Silhouette": {"metric": "euclidean"},
        "Calinski Harabasz": {},
        "Davies Bouldin": {},
    }

    ref_vals = {
        "Silhouette": 1,
        "Calinski Harabasz": np.inf,
        "Davies Bouldin": 0,
    }
    metrics = [[pf, fn(X, y_pred, **hyper[pf]), ref_vals[pf]] for pf, fn in perform.items()]

    return pd.DataFrame(metrics, columns=["Metric", "Value", "Reference"]).set_index("Metric")
