import pandas as pd

from ..utils import four_fifths_list
from ._explainability_level import explainability_ease_score
from ._importance_spread_constrast import (
    feature_importance_contrast,
    feature_importance_spread,
)
from ._surrogate_efficacy_metrics import compute_surrogate_efficacy_metrics


def fourth_fifths(feature_importance):
    feat_id = four_fifths_list(feature_importance)
    len_80 = len(feat_id)
    len_100 = len(feature_importance)
    len_80_100 = len_80 / len_100
    pfi_80_100 = pd.DataFrame({"Fourth Fifths": [len_80_100]})
    return pfi_80_100.T.rename(columns={0: "Value"})


def importance_spread_divergence(feature_importance):
    isd = feature_importance_spread(feature_importance, divergence=True)
    return isd


def importance_spread_ratio(feature_importance):
    isr = feature_importance_spread(feature_importance, divergence=False)
    return isr


def global_overlap_score(feature_importance, conditional_feature_importance):
    overlap_score = feature_importance_contrast(
        feature_importance, conditional_feature_importance
    )
    return overlap_score


def global_range_overlap_score(feature_importance, conditional_feature_importance):
    range_overlap_score = feature_importance_contrast(
        feature_importance, conditional_feature_importance, mode="range"
    )
    return range_overlap_score


def global_explainability_ease_score(model_type, model, x, y, feature_importance):
    exp_score = explainability_ease_score(model_type, model, x, y, feature_importance)
    return exp_score


def surrogate_efficacy(model_type, x, y, surrogate):
    return compute_surrogate_efficacy_metrics(model_type, x, y, surrogate)
