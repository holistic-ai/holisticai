from typing import Any, Literal

import pandas as pd
from holisticai.explainability.metrics.global_feature_importance._importance_spread import (
    FeatureImportanceSpread,
)
from holisticai.explainability.metrics.global_feature_importance._surrogate import surrogate_mean_squared_error
from holisticai.explainability.metrics.surrogate._stability import (
    FeatureImportancesStability,
    FeaturesStability,
)
from holisticai.explainability.metrics.tree._tree import (
    TreeDepthVariance,
    TreeNumberOfFeatures,
    TreeNumberOfRules,
    WeightedAverageDepth,
    WeightedAverageExplainabilityScore,
)
from holisticai.utils.surrogate_models import RegressionSurrogate


class MSEDifference:
    reference: float = 0
    name: str = "MSE Difference"

    def __call__(self, y, y_pred, y_surrogate):
        Pb = surrogate_mean_squared_error(y, y_pred)
        Pt = surrogate_mean_squared_error(y, y_surrogate)
        D = Pb - Pt
        return D


def surrogate_mean_squared_error_difference(y, y_pred, y_surrogate):
    m = MSEDifference()
    return m(y, y_pred, y_surrogate)


def regression_surrogate_explainability_metrics(
    X: Any,
    y: Any,
    y_pred: Any,
    surrogate_type: Literal["shallow_tree", "tree"],
    metric_type: Literal["performance", "stability", "tree", "all"] = "all",
    return_surrogate_model: bool = False,
):
    surrogate = RegressionSurrogate(X, y_pred=y_pred, model_type=surrogate_type)
    y_surrogate = surrogate.predict(X)

    results = {}
    is_all = metric_type == "all"
    if is_all or metric_type == "performance":
        m = MSEDifference()
        results[m.name] = {"Value": m(y, y_pred, y_surrogate), "Reference": m.reference}

        results["Surrogate MSE"] = {"Value": surrogate_mean_squared_error(y_pred, y_surrogate), "Reference": 0}

    if is_all or metric_type == "stability":
        m = FeaturesStability()
        results[m.name] = {"Value": m(X, y_pred, surrogate), "Reference": m.reference}

        m = FeatureImportancesStability()
        results[m.name] = {"Value": m(X, y_pred, surrogate), "Reference": m.reference}

        m = FeatureImportanceSpread()
        results[m.name] = {"Value": m(surrogate.feature_importances_), "Reference": m.reference}

    if is_all or metric_type == "tree":
        m = TreeNumberOfFeatures()
        results[m.name] = {"Value": m(surrogate), "Reference": m.reference}

        m = TreeNumberOfRules()
        results[m.name] = {"Value": m(surrogate), "Reference": m.reference}

        m = TreeDepthVariance()
        results[m.name] = {"Value": m(surrogate), "Reference": m.reference}

        m = WeightedAverageExplainabilityScore()
        results[m.name] = {"Value": m(surrogate), "Reference": m.reference}

        m = WeightedAverageDepth()
        results[m.name] = {"Value": m(surrogate), "Reference": m.reference}

    if return_surrogate_model:
        return pd.DataFrame(results).T, surrogate
    return pd.DataFrame(results).T
