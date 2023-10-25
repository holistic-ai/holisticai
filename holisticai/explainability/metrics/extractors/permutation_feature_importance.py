import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from ..global_importance import (
    FourthFifths,
    PositionParity, 
    RankAlignment, 
    RegionSimilarity,
    SpreadDivergence, 
    SpreadRatio,
    ExplainabilityEase
)
from ..utils import (
    BaseFeatureImportance,
    GlobalFeatureImportance,
    get_index_groups,
)


def feature_importance(model, x, y):
    n_repeats = 5
    random_state = 42
    max_samples = min(x.shape[0], 1000)
    feat_imp = permutation_importance(
        model,
        x,
        y,
        n_jobs=-1,
        n_repeats=n_repeats,
        random_state=random_state,
        max_samples=max_samples,
    )
    df_feat_imp = pd.DataFrame(
        {
            "Variable": x.columns,
            "Importance": feat_imp["importances_mean"],
            "Std": feat_imp["importances_std"],
        }
    )
    df_feat_imp["Importance"] = abs(df_feat_imp["Importance"])
    df_feat_imp["Importance"] /= df_feat_imp["Importance"].sum()
    df_feat_imp = df_feat_imp.sort_values("Importance", ascending=False).copy()

    return df_feat_imp


def compute_permutation_feature_importance(model_type, model, x, y):
    # Feature Importance
    features_importance = feature_importance(model, x, y)

    # Conditional Feature Importance (classification:category, regression:quantile)
    index_groups = get_index_groups(model_type, y)
    conditional_features_importance = {
        str(label): feature_importance(model, x.loc[index], y.loc[index])
        for label, index in index_groups.items()
    }

    return PermutationFeatureImportance(
        model_type,
        model,
        x,
        y,
        features_importance,
        conditional_features_importance,
        index_groups,
    )


class PermutationFeatureImportance(BaseFeatureImportance, GlobalFeatureImportance):
    def __init__(
        self,
        model_type,
        model,
        x,
        y,
        importance_weights,
        conditional_importance_weights,
        index_groups,
    ):
        self.model_type = model_type
        self.model = model
        self.x = x
        self.y = y
        self.feature_importance = importance_weights
        self.conditional_feature_importance = conditional_importance_weights
        self.index_groups = index_groups

    def metrics(self, alpha, detailed):

        feat_imp, (alpha_feat_imp, alpha_cond_feat_imp) = self.get_alpha_feature_importance(alpha)
        
        metrics = [SpreadDivergence(detailed=detailed), 
                   SpreadRatio(detailed=detailed), 
                   PositionParity(detailed=detailed), 
                   RankAlignment(detailed=detailed), 
                   RegionSimilarity(detailed=detailed)]
        
        ff = FourthFifths(detailed=detailed)        
        expe = ExplainabilityEase(model_type=self.model_type, model=self.model, x=self.x)
        
        
        metric_scores = []
        scores = ff(feat_imp)
        metric_scores +=[{'Metric':metric_name, 'Value':value, 'Reference': ff.reference} for metric_name,value in scores.items()]
        
        for metric_fn in metrics:
            scores = metric_fn(alpha_feat_imp, alpha_cond_feat_imp)
            metric_scores +=[{'Metric':metric_name, 'Value':value, 'Reference': metric_fn.reference} 
                             for metric_name,value in scores.items()]
                    
        scores = expe(alpha_feat_imp)
        metric_scores +=[{'Metric':metric_name, 'Value':value, 'Reference': expe.reference} for metric_name,value in scores.items()]
        
        return pd.DataFrame(metric_scores).set_index('Metric').sort_index()
