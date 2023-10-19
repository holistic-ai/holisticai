import pandas as pd

from ..utils.explainer_utils import alpha_importance_list, feature_importance_spread
from ._contrast_metrics import feature_importance_contrast
from ._explainability_level import explainability_ease_score
from ._surrogate_efficacy_metrics import compute_surrogate_efficacy_metrics


def alpha_importance(feature_importance, alpha=0.8):
    feat_id = alpha_importance_list(feature_importance, alpha)
    len_alpha = len(feat_id)
    len_100 = len(feature_importance)
    len_alpha_100 = len_alpha / len_100
    pfi_alpha_100 = pd.DataFrame({"Alpha Importance": [len_alpha_100]})
    return pfi_alpha_100.T.rename(columns={0: "Value"})


def fourth_fifths(feature_importance):
    alpha_imp = alpha_importance(feature_importance, alpha=0.8)
    alpha_imp.rename({"Alpha Importance": "Fourth Fifths"}, inplace=True)
    return alpha_imp


def importance_spread_divergence(feature_importance):
    isd = feature_importance_spread(feature_importance, divergence=True)
    return isd


def importance_spread_ratio(feature_importance):
    isr = feature_importance_spread(feature_importance, divergence=False)
    return isr


def global_overlap_score(feature_importance, conditional_feature_importance, detailed):
    overlap_score = feature_importance_contrast(
        feature_importance,
        conditional_feature_importance,
        mode="overlap",
        detailed=detailed,
    )
    return overlap_score


def global_range_overlap_score(
    feature_importance, conditional_feature_importance, detailed
):
    range_overlap_score = feature_importance_contrast(
        feature_importance,
        conditional_feature_importance,
        mode="range",
        detailed=detailed,
    )
    return range_overlap_score


def global_similarity_score(
    feature_importance, conditional_feature_importance, detailed
):
    similarity_score = feature_importance_contrast(
        feature_importance,
        conditional_feature_importance,
        mode="similarity",
        detailed=detailed,
    )
    return similarity_score


def global_explainability_ease_score(model_type, model, x, y, feature_importance):
    exp_score = explainability_ease_score(model_type, model, x, y, feature_importance)
    return exp_score


def surrogate_efficacy(model_type, x, y, surrogate):
    return compute_surrogate_efficacy_metrics(model_type, x, y, surrogate)
