from __future__ import annotations

import pandas as pd
from holisticai.explainability.metrics.tree import (
    TreeDepthVariance,
    TreeNumberOfFeatures,
    TreeNumberOfRules,
    WeightedAverageDepth,
    WeightedAverageExplainabilityScore,
    WeightedTreeGini,
)


def tree_explainability_metrics(tree) -> pd.DataFrame:
    results = []
    metric = WeightedAverageDepth()
    value = metric(tree)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    metric = WeightedAverageExplainabilityScore()
    value = metric(tree)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    metric = WeightedTreeGini()
    value = metric(tree)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    metric = TreeDepthVariance()
    value = metric(tree)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    metric = TreeNumberOfRules()
    value = metric(tree)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    metric = TreeNumberOfFeatures()
    value = metric(tree)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    return pd.DataFrame(results).set_index("metric")
