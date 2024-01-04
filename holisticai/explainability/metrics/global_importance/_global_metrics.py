import pandas as pd

from ..utils.explainer_utils import (
    SpreadDivergence,
    SpreadRatio,
    alpha_feature_importance,
)
from ._contrast_metrics import ImportantSimilarity, PositionParity, RankAlignment
from ._explainability_level import ExplainabilityEase
from ._surrogate_efficacy_metrics import SurrogacyMetric


def alpha_importance(feature_importance, alpha=0.8):
    alpha_feat_imp = alpha_feature_importance(feature_importance, alpha)
    len_alpha = len(alpha_feat_imp)
    len_100 = len(feature_importance)
    return len_alpha / len_100


class FourthFifths:
    def __init__(self, detailed):
        self.detailed = detailed
        self.reference = 0
        self.name = "Fourth Fifths"

    def __call__(self, feat_imp, cond_feat_imp=None):
        ff = {self.name: alpha_importance(feat_imp, alpha=0.8)}

        if self.detailed and (cond_feat_imp is not None):
            cond_ff = {
                f"{self.name} {k}": alpha_importance(cfi, alpha=0.8)
                for k, cfi in cond_feat_imp.items()
            }

            return {**ff, **cond_ff}

        return ff
