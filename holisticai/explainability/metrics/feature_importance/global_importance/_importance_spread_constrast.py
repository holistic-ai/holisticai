import numpy as np
import pandas as pd

from ..utils import importance_spread


def __importance_range_constrast(
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


def __importance_order_constrast(
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


def feature_importance_contrast(
    feature_importance, conditional_feature_importance, mode=None
):

    feature_importance_indexes = list(feature_importance.index)
    conditional_feature_importance_indexes = {
        k: list(v.index) for k, v in conditional_feature_importance.items()
    }

    if mode == "range":
        feature_importance_constrast = {
            f"Global Range Overlap Score {k}": __importance_range_constrast(
                feature_importance_indexes, v
            )
            for k, v in conditional_feature_importance_indexes.items()
        }

    else:
        feature_importance_constrast = {
            f"Global Overlap Score {k}": __importance_order_constrast(
                feature_importance_indexes, v
            )
            for k, v in conditional_feature_importance_indexes.items()
        }

    return pd.DataFrame.from_dict(
        feature_importance_constrast, orient="index", columns=["Value"]
    )


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
