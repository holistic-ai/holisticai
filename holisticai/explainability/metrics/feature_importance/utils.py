import numpy as np
import pandas as pd
from lime import lime_tabular
from sklearn.inspection import PartialDependenceDisplay


def four_fifths_list(feature_importance, cutoff=None):
    """
    Parameters
    ----------
    feature_importance: np.array
        array with raw feature importance
    cutoff: float
        threshold for feature importance
    """
    if cutoff is None:
        cutoff = 0.80

    importance = feature_importance["Importance"]
    feature_names = feature_importance["Variable"]

    feature_weight = importance / sum(importance)

    # entropy or divergence
    return feature_names.loc[(feature_weight.cumsum() < cutoff).values]


def feature_importance_spread(
    features_importance, conditional_features_importance=None, divergence=False
):
    spread_type = "Divergence" if divergence else "Ratio"

    feat_importance_spread = {
        f"Importance Spread {spread_type}": [
            importance_spread(features_importance["Importance"], divergence=divergence)
        ]
    }

    if conditional_features_importance is not None:

        feat_importance_spread.update(
            {
                f"Conditional Importance Spread {spread_type}[{c}]": [
                    importance_spread(importance["Importance"], divergence=divergence)
                ]
                for c, importance in conditional_features_importance.items()
            }
        )

    imp_spread = pd.DataFrame(feat_importance_spread)

    return imp_spread.T.rename(columns={0: "Value"})


def importance_spread(feature_importance, divergence=False):
    """
    Parameters
    ----------
    feature_importance: np.array
        array with raw feature importance
    divergence: bool
        if True calculate divergence instead of ratio
    """
    if len(feature_importance) == 0:
        return 0 if divergence else 1

    importance = feature_importance
    from scipy.stats import entropy

    feature_weight = importance / sum(importance)
    feature_equal_weight = np.array([1.0 / len(importance)] * len(importance))

    # entropy or divergence
    if divergence is False:
        return entropy(feature_weight) / entropy(feature_equal_weight)  # ratio
    else:
        return entropy(feature_weight, feature_equal_weight)  # divergence


def explanation_contrast(feature_importance, cond_feature_importance, order=True):
    """
    Parameters
    ----------
    feature_importance: np.array
        array with raw feature importance
    cond_feature_importance: np.array
        array with raw feature importance
    order: bool
        if True calculate order contrast
    """
    # todo - implement per weight
    # in case we are per measuring order
    if order:
        conditionals = list(cond_feature_importance.keys())
        results = []
        for c in conditionals:
            matching = (
                feature_importance["Variable"].index
                == cond_feature_importance[c]["Variable"].index
            )
            results += (c, matching.mean())

    return results


def importance_range_constrast(
    feature_importance_indexes: np.ndarray,
    conditional_feature_importance_indexes: np.ndarray,
):
    """
    Parameters
    ----------
    feature_importance_indexes: np.array
        array with feature importance indexes
    conditional_feature_importance_indexes: np.array
        array with conditional feature importance indexes
    """
    m_range = []
    for top_k in range(1, len(feature_importance_indexes) + 1):
        ggg = set(feature_importance_indexes[:top_k])
        vvv = set(conditional_feature_importance_indexes[:top_k])
        u = len(set(ggg).intersection(vvv)) / top_k
        m_range.append(u)
    m_range = np.array(m_range)

    return m_range.mean()


def importance_order_constrast(
    feature_importance_indexes: np.ndarray,
    conditional_features_importance_indexes: np.ndarray,
):
    """
    Parameters
    ----------
    feature_importance_indexes: np.array
        array with feature importance indexes
    conditional_feature_importance_indexes: np.array
        array with conditional feature importance indexes
    """
    m_order = np.array(feature_importance_indexes) == np.array(
        conditional_features_importance_indexes
    )
    m_order = np.cumsum(m_order) / np.arange(1, len(m_order) + 1)

    return m_order.mean()


def partial_dependence_creator(model, grid_resolution, x, feature_ids, target=None):
    """
    Parameters
    ----------
    model: sklearn-like object
        sklearn-like object with predict method
    grid_resolution: int
        number of points to compute partial dependence
    X: np.array
        input data
    feature_ids: list
        list of feature ids
    target: int
        target class
    """
    # to do
    # -> explicit implementation of plot_partial_dependence_plot from sklearn without showing the chart
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["interactive"] == False

    feature_names = list(x.columns)
    method = "brute"
    percentiles = (0.05, 0.95)
    response_method = "auto"

    kargs = {
        "estimator": model,
        "X": x,
        "features": feature_ids,
        "feature_names": feature_names,
        "percentiles": percentiles,
        "response_method": response_method,
        "method": method,
        "grid_resolution": grid_resolution,
        "n_jobs": -1,
        "subsample": 100,
    }

    if not target == None:
        kargs.update({"target": target})

    g = PartialDependenceDisplay.from_estimator(**kargs)
    plt.close()

    pd_results = {}
    for (i, f) in enumerate(feature_ids):
        pd_results[feature_names[f]] = pd.DataFrame(
            {
                "score": g.pd_results[i]["average"][0],
                "values": g.pd_results[i]["values"][0],
            }
        )

    return pd_results
