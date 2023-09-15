import pandas as pd

from holisticai.explainability.metrics.feature_importance.utils import importance_spread


def feature_importance_spread_lime(
    feature_importance, conditional_feature_importance, lime_importance
):
    divergence = False
    if lime_importance == "dataset":
        metric_name = "Dataset"
        imp_spread = {
            "Global": feature_importance.groupby("Sample Id").apply(
                lambda df: importance_spread(
                    df.set_index("Feature Label")["Importance"], divergence=divergence
                )
            )
        }
        for c, ccfi in conditional_feature_importance.items():
            imp_spread.update(
                {
                    str(c): ccfi.groupby("Sample Id").apply(
                        lambda df: importance_spread(
                            df.set_index("Feature Label")["Importance"],
                            divergence=divergence,
                        )
                    )
                }
            )
    else:
        metric_name = "Features"
        imp_spread = {
            "Global": feature_importance.groupby("Feature Label")["Importance"].apply(
                lambda x: importance_spread(x, divergence=divergence)
            )
        }
        for c, df_cls in conditional_feature_importance.items():
            imp_spread[str(c)] = df_cls.groupby("Feature Label")["Importance"].apply(
                lambda x: importance_spread(x, divergence=divergence)
            )

    spread_imp_divergence = {
        k: importance_spread(v, divergence=True) for k, v in imp_spread.items()
    }
    spread_imp_ratio = {
        k: importance_spread(v, divergence=False) for k, v in imp_spread.items()
    }
    mean_imp_spread = {k: v.mean() for k, v in imp_spread.items()}

    set_name = lambda x, c: x if c == "Global" else f"{x} {c}"
    table = {
        set_name(f"{metric_name} Spread Stability", c): [v]
        for c, v in spread_imp_divergence.items()
    }
    table.update(
        {
            set_name(f"{metric_name} Spread Mean", c): [v]
            for c, v in mean_imp_spread.items()
        }
    )
    table.update(
        {
            set_name(f"{metric_name} Spread Ratio", c): [v]
            for c, v in spread_imp_ratio.items()
        }
    )
    table = pd.DataFrame(table)

    result = {
        f"{metric_name} Spread Divergence": spread_imp_divergence["Global"],
        f"{metric_name} Spread Mean": mean_imp_spread["Global"],
        f"{metric_name} Spread Stability": spread_imp_ratio["Global"],
    }

    result = pd.DataFrame(result, index=[0])

    return {
        "table": table.T.rename(columns={0: "Value"}),
        "result": result.T.rename(columns={0: "Value"}),
        "imp_spread": imp_spread,
    }
