from __future__ import annotations

from typing import Union

import pandas as pd
from holisticai.explainability.metrics.global_importance import (
    AlphaScore,
    PositionParity,
    RankAlignment,
    SpreadDivergence,
    SpreadRatio,
    XAIEaseScore,
)
from holisticai.explainability.metrics.global_importance._surrogate import surrogate_accuracy_score
from holisticai.explainability.metrics.local_importance import DataStability, FeatureStability
from holisticai.utils import (
    ConditionalImportances,
    Importances,
    LocalImportances,
    PartialDependence,
)


def multiclass_explainability_metrics(
    importances: Importances,
    partial_dependencies: PartialDependence,
    conditional_importances: ConditionalImportances,
    X: Union[pd.DataFrame, None] = None,
    y_pred: Union[pd.Series, None] = None,
    local_importances: Union[LocalImportances, None] = None,
) -> pd.DataFrame:
    ranked_importances = importances.top_alpha(0.8)
    results = []
    metric = AlphaScore()
    value = metric(importances)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    metric = XAIEaseScore()
    value = metric(partial_dependencies, ranked_importances)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    metric = PositionParity()
    value = metric(conditional_importances, ranked_importances)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    metric = RankAlignment()
    value = metric(conditional_importances, ranked_importances)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    metric = SpreadRatio()
    value = metric(importances)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    metric = SpreadDivergence()
    value = metric(importances)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    if "surrogate" in importances.extra_attrs:
        y_surrogate = importances.extra_attrs["surrogate"].predict(X)
        value = surrogate_accuracy_score(y_pred, y_surrogate)
        results.append({"metric": "Surrogate Accuracy Score", "value": value, "reference": 1})

    if local_importances is not None:
        metric = DataStability()
        value = metric(local_importances)
        results.append({"metric": metric.name, "value": value, "reference": metric.reference})

        metric = FeatureStability()
        value = metric(local_importances)
        results.append({"metric": metric.name, "value": value, "reference": metric.reference})
    return pd.DataFrame(results).set_index("metric")
