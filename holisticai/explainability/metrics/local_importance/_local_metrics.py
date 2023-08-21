from holisticai.utils.models.feature_importance.local_metrics.lime_importance_spread import feature_importance_spread_lime
import pandas as pd

def dataset_spread_stability(feature_importance, conditional_feature_importance):
    dfis = feature_importance_spread_lime(feature_importance, conditional_feature_importance, lime_importance = 'dataset')
    return dfis

def features_spread_stability(feature_importance, conditional_feature_importance):
    fdis = feature_importance_spread_lime(feature_importance, conditional_feature_importance, lime_importance = 'features')
    return fdis