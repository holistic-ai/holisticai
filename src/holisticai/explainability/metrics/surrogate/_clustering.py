from typing import Any, Literal

import pandas as pd
from sklearn.metrics import accuracy_score

from holisticai.explainability.metrics.global_feature_importance._importance_spread import (
    FeatureImportanceSpread,
)
from holisticai.explainability.metrics.global_feature_importance._surrogate import (
    surrogate_accuracy_score,
)
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
    WeightedTreeGini,
)
from holisticai.utils.surrogate_models import ClusteringSurrogate


class AccuracyDegradation:
    reference: float = 0
    name: str = "Accuracy Difference"

    def __call__(self, X, y, proxy, surrogate):
        y_pred = proxy.predict(X)
        Pb = accuracy_score(y, y_pred)
        y_pred_surrogate = surrogate.predict(X)
        Pt = accuracy_score(y, y_pred_surrogate)
        return Pb - Pt


def accuracy_difference(X, y, proxy, surrogate):
    m = AccuracyDegradation()
    return m(X, y, proxy, surrogate)


def clustering_surrogate_explainability_metrics(
    X: Any,
    labels: Any,
    surrogate_type: Literal["shallow_tree", "tree"],
    metric_type: Literal["performance", "stability", "tree", "all"] = "all",
    return_surrogate_model: bool = False,
):
    surrogate = ClusteringSurrogate(X, labels, model_type=surrogate_type)
    results = {}
    is_all = metric_type == "all"
    if is_all or metric_type == "performance":
        results["Surrogate Accuracy"] = {
            "Value": surrogate_accuracy_score(labels, surrogate.predict(X)),
            "Reference": 1,
        }

    if is_all or metric_type == "stability":
        m = FeaturesStability()
        results[m.name] = {"Value": m(X, y_pred=labels, surrogate=surrogate), "Reference": m.reference}

        m = FeatureImportancesStability()
        results[m.name] = {"Value": m(X, y_pred=labels, surrogate=surrogate), "Reference": m.reference}

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

        m = WeightedTreeGini()
        results[m.name] = {"Value": m(surrogate), "Reference": m.reference}

    if return_surrogate_model:
        return pd.DataFrame(results).T, surrogate
    return pd.DataFrame(results).T
