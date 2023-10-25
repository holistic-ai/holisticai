from holisticai.explainability.metrics.utils.explainer_utils import (
    gini_coefficient,
    importance_spread,
)

"""
def importance_distribution_variation(importance, mode):
    if mode == "gini":
        return gini_coefficient(importance)

    elif mode == "divergence":
        return importance_spread(importance, divergence=True)

    elif mode == "ratio":
        return importance_spread(importance, divergence=False)

    else:
        raise (f"Unknown distribution variation type: {mode}")
"""


class FeatureStability:
    def __init__(self, detailed):
        self.name = "Feature Stability"
        self.reference = 0
        self.detailed = detailed

    def __call__(self, feature_importance, conditional_feature_importance, reduce=True):
        def spread_ratio_function(importance):
            return importance_spread(importance, divergence=False)

        imp_spread = {
            self.name: feature_importance.groupby("Feature Label")["Importance"].apply(
                spread_ratio_function
            )
        }

        if self.detailed:
            for c, cfi in conditional_feature_importance.items():
                imp_spread[f"{self.name} {c}"] = cfi.groupby("Feature Label")[
                    "Importance"
                ].apply(spread_ratio_function)

        if reduce:
            return {k: gini_coefficient(v) for k, v in imp_spread.items()}
        else:
            return imp_spread


class DataStability:
    def __init__(self, detailed):
        self.name = "Data Stability"
        self.reference = 0
        self.detailed = detailed

    def __call__(self, feature_importance, conditional_feature_importance, reduce=True):
        def spread_ratio_function(df):
            importance = df.set_index("Feature Label")["Importance"]
            return importance_spread(importance, divergence=False)

        imp_spread = {
            self.name: feature_importance.groupby("Sample Id").apply(
                spread_ratio_function
            )
        }

        if self.detailed:
            for c, ccfi in conditional_feature_importance.items():
                imp_spread[f"{self.name} {c}"] = ccfi.groupby("Sample Id").apply(
                    spread_ratio_function
                )

        if reduce:
            return {k: gini_coefficient(v) for k, v in imp_spread.items()}
        else:
            return imp_spread


"""
def feature_importance_spread_lime(
    feature_importance, conditional_feature_importance, lime_importance
):
    metric_name = "Features"
    imp_spread = {
        "Global": feature_importance.groupby("Feature Label")["Importance"].apply(
            lambda x: importance_distribution_variation(x)
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
"""
