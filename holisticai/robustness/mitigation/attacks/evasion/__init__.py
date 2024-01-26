"""
Module providing evasion attacks under a common interface.
"""
from holisticai.robustness.mitigation.attacks.evasion.adversarial_asr import (
    CarliniWagnerASR,
)
from holisticai.robustness.mitigation.attacks.evasion.adversarial_patch.adversarial_patch import (
    AdversarialPatch,
)
from holisticai.robustness.mitigation.attacks.evasion.adversarial_patch.adversarial_patch_numpy import (
    AdversarialPatchNumpy,
)
from holisticai.robustness.mitigation.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import (
    AdversarialPatchPyTorch,
)
from holisticai.robustness.mitigation.attacks.evasion.adversarial_patch.adversarial_patch_tensorflow import (
    AdversarialPatchTensorFlowV2,
)
from holisticai.robustness.mitigation.attacks.evasion.adversarial_texture.adversarial_texture_pytorch import (
    AdversarialTexturePyTorch,
)
from holisticai.robustness.mitigation.attacks.evasion.auto_attack import AutoAttack
from holisticai.robustness.mitigation.attacks.evasion.auto_projected_gradient_descent import (
    AutoProjectedGradientDescent,
)
from holisticai.robustness.mitigation.attacks.evasion.boundary import BoundaryAttack
from holisticai.robustness.mitigation.attacks.evasion.brendel_bethge import (
    BrendelBethgeAttack,
)
from holisticai.robustness.mitigation.attacks.evasion.carlini import (
    CarliniL0Method,
    CarliniL2Method,
    CarliniLInfMethod,
)
from holisticai.robustness.mitigation.attacks.evasion.decision_tree_attack import (
    DecisionTreeAttack,
)
from holisticai.robustness.mitigation.attacks.evasion.deepfool import DeepFool
from holisticai.robustness.mitigation.attacks.evasion.dpatch import DPatch
from holisticai.robustness.mitigation.attacks.evasion.dpatch_robust import RobustDPatch
from holisticai.robustness.mitigation.attacks.evasion.elastic_net import ElasticNet
from holisticai.robustness.mitigation.attacks.evasion.fast_gradient import (
    FastGradientMethod,
)
from holisticai.robustness.mitigation.attacks.evasion.feature_adversaries.feature_adversaries_numpy import (
    FeatureAdversariesNumpy,
)
from holisticai.robustness.mitigation.attacks.evasion.feature_adversaries.feature_adversaries_pytorch import (
    FeatureAdversariesPyTorch,
)
from holisticai.robustness.mitigation.attacks.evasion.feature_adversaries.feature_adversaries_tensorflow import (
    FeatureAdversariesTensorFlowV2,
)
from holisticai.robustness.mitigation.attacks.evasion.frame_saliency import (
    FrameSaliencyAttack,
)
from holisticai.robustness.mitigation.attacks.evasion.geometric_decision_based_attack import (
    GeoDA,
)
from holisticai.robustness.mitigation.attacks.evasion.graphite.graphite_blackbox import (
    GRAPHITEBlackbox,
)
from holisticai.robustness.mitigation.attacks.evasion.graphite.graphite_whitebox_pytorch import (
    GRAPHITEWhiteboxPyTorch,
)
from holisticai.robustness.mitigation.attacks.evasion.hclu import (
    HighConfidenceLowUncertainty,
)
from holisticai.robustness.mitigation.attacks.evasion.hop_skip_jump import HopSkipJump

# from holisticai.robustness.mitigation.attacks.evasion.imperceptible_asr.imperceptible_asr import ImperceptibleASR
# from holisticai.robustness.mitigation.attacks.evasion.imperceptible_asr.imperceptible_asr_pytorch import ImperceptibleASRPyTorch
from holisticai.robustness.mitigation.attacks.evasion.iterative_method import (
    BasicIterativeMethod,
)
from holisticai.robustness.mitigation.attacks.evasion.laser_attack.laser_attack import (
    LaserAttack,
)
from holisticai.robustness.mitigation.attacks.evasion.lowprofool import LowProFool
from holisticai.robustness.mitigation.attacks.evasion.momentum_iterative_method import (
    MomentumIterativeMethod,
)
from holisticai.robustness.mitigation.attacks.evasion.newtonfool import NewtonFool
from holisticai.robustness.mitigation.attacks.evasion.over_the_air_flickering.over_the_air_flickering_pytorch import (
    OverTheAirFlickeringPyTorch,
)
from holisticai.robustness.mitigation.attacks.evasion.pe_malware_attack import (
    MalwareGDTensorFlow,
)
from holisticai.robustness.mitigation.attacks.evasion.pixel_threshold import (
    PixelAttack,
    ThresholdAttack,
)
from holisticai.robustness.mitigation.attacks.evasion.projected_gradient_descent.projected_gradient_descent import (
    ProjectedGradientDescent,
)
from holisticai.robustness.mitigation.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
    ProjectedGradientDescentNumpy,
)
from holisticai.robustness.mitigation.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import (
    ProjectedGradientDescentPyTorch,
)
from holisticai.robustness.mitigation.attacks.evasion.projected_gradient_descent.projected_gradient_descent_tensorflow_v2 import (
    ProjectedGradientDescentTensorFlowV2,
)
from holisticai.robustness.mitigation.attacks.evasion.saliency_map import (
    SaliencyMapMethod,
)
from holisticai.robustness.mitigation.attacks.evasion.shadow_attack import ShadowAttack
from holisticai.robustness.mitigation.attacks.evasion.sign_opt import SignOPTAttack

# from holisticai.robustness.mitigation.attacks.evasion.shapeshifter import ShapeShifter
from holisticai.robustness.mitigation.attacks.evasion.simba import SimBA
from holisticai.robustness.mitigation.attacks.evasion.spatial_transformation import (
    SpatialTransformation,
)
from holisticai.robustness.mitigation.attacks.evasion.square_attack import SquareAttack
from holisticai.robustness.mitigation.attacks.evasion.targeted_universal_perturbation import (
    TargetedUniversalPerturbation,
)
from holisticai.robustness.mitigation.attacks.evasion.universal_perturbation import (
    UniversalPerturbation,
)
from holisticai.robustness.mitigation.attacks.evasion.virtual_adversarial import (
    VirtualAdversarialMethod,
)
from holisticai.robustness.mitigation.attacks.evasion.wasserstein import Wasserstein
from holisticai.robustness.mitigation.attacks.evasion.zoo import ZooAttack

from ._constants import (
    DECISION_TREE_ATTACKERS,
    PYTORCH_ATTACKERS,
    SKLEARN_ATTACKERS,
    Attacker,
)
