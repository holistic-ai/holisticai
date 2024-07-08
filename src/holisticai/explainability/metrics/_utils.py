import pandas as pd
from holisticai.explainability.metrics.global_importance import (
    AlphaImportanceScore,
    PositionParity,
    RankAlignment,
    SpreadRatio,
    SpreadDivergence,
    XAIEaseScore,
)
from holisticai.explainability.metrics.local_importance import DataStability, FeatureStability


def compute_global_explainability_metrics_from_features(xai_features):
    results = []
    metric = AlphaImportanceScore()
    value = metric(xai_features.feature_importance, xai_features.ranked_feature_importance)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    metric = XAIEaseScore()
    value = metric(xai_features.partial_dependence, xai_features.ranked_feature_importance)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    metric = PositionParity()
    value = metric(xai_features.conditional_feature_importance, xai_features.ranked_feature_importance)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    metric = RankAlignment()
    value = metric(xai_features.conditional_feature_importance, xai_features.ranked_feature_importance)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    metric = SpreadRatio()
    value = metric(xai_features.feature_importance)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    metric = SpreadDivergence()
    value = metric(xai_features.feature_importance)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})
    
    return pd.DataFrame(results).set_index("metric")


def compute_local_explainability_metrics_from_features(xai_features):
    results = []
    metric = FeatureStability()
    value = metric(xai_features.local_feature_importance)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})

    metric = DataStability()
    value = metric(xai_features.local_feature_importance)
    results.append({"metric": metric.name, "value": value, "reference": metric.reference})
    return pd.DataFrame(results).set_index("metric")


def compute_explainability_metrics_from_features(xai_features, metric_type="global"):
    if metric_type == "global":
        return compute_global_explainability_metrics_from_features(xai_features)

    if metric_type == "local":
        if xai_features.local_feature_importance is None:
            raise ValueError("Local feature importance can't be none for local metrics.")
        return compute_local_explainability_metrics_from_features(xai_features)

    if metric_type == "both":
        if xai_features.local_feature_importance is None:
            raise ValueError("Local feature importance can't be none for local metrics.")
        global_metrics = compute_global_explainability_metrics_from_features(xai_features)
        local_metrics = compute_local_explainability_metrics_from_features(xai_features)
        return pd.concat([global_metrics, local_metrics], axis=0)

    raise ValueError(f"Invalid metric_type: {metric_type}. Must be one of 'global', 'local', 'both'.")
