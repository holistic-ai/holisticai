
import pandas as pd
from holisticai.xai.metrics.global_importance import AlphaImportanceScore, PositionParity, RankAlignment, XAIEaseScore


def compute_xai_metrics_from_features(xai_features):
    results = []
    metric = AlphaImportanceScore()
    value = metric(xai_features.feature_importance, xai_features.ranked_feature_importance)
    results.append({'metric': metric.name, 'value': value, 'reference': metric.reference})

    metric = XAIEaseScore()
    value = metric(xai_features.partial_dependence, xai_features.ranked_feature_importance)
    results.append({'metric': metric.name, 'value': value, 'reference': metric.reference})

    metric = PositionParity()
    value = metric(xai_features.conditional_feature_importance, xai_features.ranked_feature_importance)
    results.append({'metric': metric.name, 'value': value, 'reference': metric.reference})

    metric = RankAlignment()
    value = metric(xai_features.conditional_feature_importance, xai_features.ranked_feature_importance)
    results.append({'metric': metric.name, 'value': value, 'reference': metric.reference})
    return pd.DataFrame(results).set_index("metric")
