import numpy as np
import pandas as pd

from holisticai.explainability.metrics.utils import (
    importance_order_constrast,
    importance_range_constrast,
    importance_spread,
)


def feature_importance_contrast(
    feature_importance, conditional_feature_importance, mode=None
):

    feature_importance_indexes = list(feature_importance["Variable"].index)
    conditional_feature_importance_indexes = {
        k: list(v["Variable"].index) for k, v in conditional_feature_importance.items()
    }

    if mode == "range":
        feature_importance_constrast = {
            f"Global Range Overlap Score {k}": importance_range_constrast(
                feature_importance_indexes, v
            )
            for k, v in conditional_feature_importance_indexes.items()
        }

    else:
        feature_importance_constrast = {
            f"Global Overlap Score {k}": importance_order_constrast(
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
    SPREAD_TYPE = "Divergence" if divergence else "Ratio"

    feat_importance_spread = {
        f"Importance Spread {SPREAD_TYPE}": [
            importance_spread(features_importance["Importance"], divergence=divergence)
        ]
    }

    if conditional_features_importance is not None:

        feat_importance_spread.update(
            {
                f"Conditional Importance Spread {SPREAD_TYPE}[{c}]": [
                    importance_spread(importance["Importance"], divergence=divergence)
                ]
                for c, importance in conditional_features_importance.items()
            }
        )

    imp_spread = pd.DataFrame(feat_importance_spread)

    return imp_spread.T.rename(columns={0: "Value"})
