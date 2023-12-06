import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay

from holisticai.utils._validation import (
    _array_like_to_series,
    _matrix_like_to_dataframe,
)


def check_alpha_domain(alpha):
    if alpha is not None:
        assert (alpha >= 0) and (
            alpha <= 1
        ), f"alpha must be between 0 and 1. Valor found: {alpha}"


def check_feature_importance(x, y=None, values=None):
    if not isinstance(x, pd.DataFrame):
        x = _matrix_like_to_dataframe(x)

    x = x.astype(float)

    if (y is None) and (values is None):
        return x

    if not isinstance(y, pd.Series):
        y = _array_like_to_series(y)

    if not y.index.equals(x):
        y.index = x.index

    if values is None:
        return x, y
    else:
        if not isinstance(x, pd.DataFrame):
            values = pd.DataFrame(values, columns=x.columns, index=x.index)
        else:
            values.index = x.index
        return x, y, values


def alpha_feature_importance(feature_importance, alpha=None):
    """
    Parameters
    ----------
    feature_importance: np.array
        array with raw feature importance
    alpha: float
        threshold for feature importance
    """
    if alpha is None:
        alpha = 0.8
    
    feature_importance = feature_importance.sort_values("Importance", ascending=False)

    importance = feature_importance["Importance"]

    feature_weight = importance / sum(importance)

    accum_feature_weight = feature_weight.cumsum()
    # entropy or divergence
    return feature_importance.loc[accum_feature_weight < alpha]


def alpha_importance_list_lime(
    feature_importance, feature_importance_names, alpha=None
):
    """
    Parameters
    ----------
    feature_importance: np.array
        array with raw feature importance
    feature_importance_names: list
        list with names
    alpha: float
        threshold for feature importance
    """
    if alpha is None:
        alpha = 0.80

    feature_weight = feature_importance / sum(feature_importance)
    return feature_importance_names.loc[(feature_weight.cumsum() < alpha).values]


class Spread:
    def __init__(self, divergence, detailed):
        self.divergence = divergence
        self.detailed = detailed

    def __call__(self, feat_imp, cond_feat_imp):
        spread = {
            self.name: importance_spread(
                feat_imp["Importance"], divergence=self.divergence
            )
        }

        if self.detailed and (cond_feat_imp is not None):

            cond_spread = {}
            for c, cfi in cond_feat_imp.items():
                cond_spread[f"{self.name} {c}"] = importance_spread(
                    cfi["Importance"], divergence=self.divergence
                )

            return {**spread, **cond_spread}

        return spread


class SpreadDivergence(Spread):
    def __init__(self, detailed):
        super().__init__(divergence=True, detailed=detailed)
        self.name = "Spread Divergence"
        self.reference = "-"


class SpreadRatio(Spread):
    def __init__(self, detailed):
        super().__init__(divergence=False, detailed=detailed)
        self.name = "Spread Ratio"
        self.reference = 0


def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x) ** 2 * np.mean(x))


def importance_spread(feature_importance, divergence=False):
    """
    Parameters
    ----------
    feature_importance: np.array
        array with raw feature importance
    divergence: bool
        if True calculate divergence instead of ratio
    """
    if len(feature_importance) == 0 or sum(feature_importance) < 1e-8:
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


def partial_dependence_creator(
    model, grid_resolution, x, features, target=None, random_state=42
):
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
    method = "auto"

    response_method = "auto"

    kargs = {
        "estimator": model,
        "X": x,
        "features": features,
        "feature_names": feature_names,
        "response_method": response_method,
        "method": method,
        "grid_resolution": grid_resolution,
        "n_jobs": -1,
        "subsample": 100,
    }

    if target is None:
        kargs.update({"percentiles": (0.05, 0.95)})
    else:
        kargs.update({"target": target, "percentiles": (0, 1)})

    g = PartialDependenceDisplay.from_estimator(**kargs)

    plt.close()

    pd_results = {}
    for (i, f) in enumerate(features):
        pd_results[f] = pd.DataFrame(
            {
                "score": g.pd_results[i]["average"][0],
                "values": g.pd_results[i]["values"][0],
            }
        )

    return pd_results
