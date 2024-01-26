"""
Module providing membership inference attacks.
"""
from .black_box import MembershipInferenceBlackBox
from .black_box_rule_based import MembershipInferenceBlackBoxRuleBased
from .label_only_boundary_distance import LabelOnlyDecisionBoundary
from .label_only_gap_attack import LabelOnlyGapAttack
from .shadow_models import ShadowModels
