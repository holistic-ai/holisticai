import numpy as np
import pandas as pd

from holisticai.explainability.metrics.utils.explainer_utils import (
    gini_coefficient,
    importance_spread,
)


def importance_distribution_variation(importance, mode):
    if mode == "gini":
        return gini_coefficient(importance)

    elif mode == "divergence":
        return importance_spread(importance, divergence=True)

    elif mode == "ratio":
        return importance_spread(importance, divergence=False)

    else:
        raise (f"Unknown distribution variation type: {mode}")


def feature_importance_spread_lime(
    feature_importance, conditional_feature_importance, lime_importance
):
    mode = "ratio"
    if lime_importance == "dataset":
        metric_name = "Dataset"
        imp_spread = {
            "Global": feature_importance.groupby("Sample Id").apply(
                lambda df: importance_distribution_variation(
                    df.set_index("Feature Label")["Importance"], mode=mode
                )
            )
        }
        for c, ccfi in conditional_feature_importance.items():
            imp_spread.update(
                {
                    str(c): ccfi.groupby("Sample Id").apply(
                        lambda df: importance_distribution_variation(
                            df.set_index("Feature Label")["Importance"], mode=mode
                        )
                    )
                }
            )
    else:
        metric_name = "Features"
        imp_spread = {
            "Global": feature_importance.groupby("Feature Label")["Importance"].apply(
                lambda x: importance_distribution_variation(x, mode=mode)
            )
        }
        for c, df_cls in conditional_feature_importance.items():
            imp_spread[str(c)] = df_cls.groupby("Feature Label")["Importance"].apply(
                lambda x: importance_distribution_variation(x, mode=mode)
            )

    spread_imp_ratio = {
        k: importance_distribution_variation(v, mode="ratio")
        for k, v in imp_spread.items()
    }
    spread_imp_gini = {
        k: importance_distribution_variation(v, mode="gini")
        for k, v in imp_spread.items()
    }
    mean_imp_spread = {k: v.mean() for k, v in imp_spread.items()}

    result = {
        f"{metric_name} Stability Gini": spread_imp_gini,
        # f"{metric_name} Stability Mean": mean_imp_spread,
        # f"{metric_name} Stability Ratio": spread_imp_ratio,
    }

    return {
        "result": result,
        "imp_spread": imp_spread,
    }
