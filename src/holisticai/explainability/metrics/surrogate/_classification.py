from typing import Any, Literal

import numpy as np
import pandas as pd
from holisticai.explainability.metrics.global_feature_importance._importance_spread import FeatureImportanceSpread
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
from holisticai.utils.surrogate_models import BinaryClassificationSurrogate, MultiClassificationSurrogate
from sklearn.metrics import accuracy_score


class AccuracyDifference:
    reference: float = 0
    name: str = "Surrogate Accuracy Difference"

    def __call__(self, y, y_pred, y_surrogate):
        Pb = accuracy_score(y, y_pred)
        Pt = accuracy_score(y, y_surrogate)
        D = Pb - Pt
        return D


def surrogate_accuracy_difference(y, y_pred, y_surrogate):
    m = AccuracyDifference()
    return m(y, y_pred, y_surrogate)


def classification_surrogate_explainability_metrics(
    X: Any,
    y: Any,
    y_pred: Any,
    surrogate_type: Literal["shallow_tree", "tree"],
    metric_type: Literal["performance", "stability", "tree", "all"] = "all",
    return_surrogate_model: bool = False,
):
    if len(np.unique(y_pred)) == 2:
        surrogate = BinaryClassificationSurrogate(X, y_pred=y_pred, model_type=surrogate_type)
    elif len(np.unique(y_pred)) > 2:
        surrogate = MultiClassificationSurrogate(X, y_pred=y_pred, model_type=surrogate_type)
    else:
        raise ValueError("y_pred must have at least two unique values")

    y_surrogate = surrogate.predict(X)
    results = {}
    is_all = metric_type == "all"
    if is_all or metric_type == "performance":
        m = AccuracyDifference()
        results[m.name] = {"Value": m(y, y_pred, y_surrogate), "Reference": m.reference}

        results["Surrogate Accuracy"] = {"Value": surrogate_accuracy_score(y_pred, y_surrogate), "Reference": 1}

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

        m = WeightedTreeGini()
        results[m.name] = {"Value": m(surrogate), "Reference": m.reference}

    if return_surrogate_model:
        return pd.DataFrame(results).T, surrogate
    return pd.DataFrame(results).T
