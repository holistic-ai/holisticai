"""
Module providing adversarial attacks under a common interface.
"""
from .attack import (
    Attack,
    AttributeInferenceAttack,
    EvasionAttack,
    ExtractionAttack,
    InferenceAttack,
    PoisoningAttack,
    PoisoningAttackBlackBox,
    PoisoningAttackTransformer,
    PoisoningAttackWhiteBox,
    ReconstructionAttack,
)

# from .attacks import evasion
# from .attacks import extraction
# from .attacks import inference
# from .attacks import poisoning

__all__ = [
    "Attack",
    "EvasionAttack",
    "PoisoningAttack",
    "PoisoningAttackBlackBox",
    "PoisoningAttackWhiteBox",
    "PoisoningAttackTransformer",
    "ExtractionAttack",
    "InferenceAttack",
    "AttributeInferenceAttack",
    "ReconstructionAttack",
]
