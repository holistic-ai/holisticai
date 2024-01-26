"""
Module providing attribute inference attacks.
"""
from .baseline import AttributeInferenceBaseline
from .black_box import AttributeInferenceBlackBox
from .meminf_based import AttributeInferenceMembership
from .true_label_baseline import AttributeInferenceBaselineTrueLabel
from .white_box_decision_tree import AttributeInferenceWhiteBoxDecisionTree
from .white_box_lifestyle_decision_tree import (
    AttributeInferenceWhiteBoxLifestyleDecisionTree,
)
