"""
Module providing attribute inference attacks.
"""
from .attribute_inference.baseline import AttributeInferenceBaseline
from .attribute_inference.black_box import AttributeInferenceBlackBox
from .attribute_inference.meminf_based import AttributeInferenceMembership
from .attribute_inference.true_label_baseline import AttributeInferenceBaselineTrueLabel
from .attribute_inference.white_box_decision_tree import (
    AttributeInferenceWhiteBoxDecisionTree,
)
from .attribute_inference.white_box_lifestyle_decision_tree import (
    AttributeInferenceWhiteBoxLifestyleDecisionTree,
)
