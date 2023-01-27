# imports
from .calibrated_eq_odds_postprocessing import CalibratedEqualizedOdds
from .debiasing_exposure.transformer import DebiasingExposure
from .eq_odds_postprocessing import EqualizedOdds
from .fair_topk.transformer import FairTopK
from .lp_debiaser.binary_balancer.transformer import LPDebiaserBinary
from .lp_debiaser.multiclass_balancer.transformer import LPDebiaserMulticlass
from .ml_debiaser.transformer import MLDebiaser
from .plugin_estimator_and_recalibration.transformer import (
    PluginEstimationAndCalibration,
)
from .reject_option_classification import RejectOptionClassification
from .wasserstein_barycenters.transformer import WassersteinBarycenter

__all__ = [
    "CalibratedEqualizedOdds",
    "EqualizedOdds",
    "RejectOptionClassification",
    "WassersteinBarycenter",
    "PluginEstimationAndCalibration",
    "MLDebiaser",
    "LPDebiaserBinary",
    "LPDebiaserMulticlass",
    "DebiasingExposure",
    "FairTopK",
]

import importlib

networkx_spec = importlib.util.find_spec("networkx")
if networkx_spec is not None:
    from .disparate_impact_remover_rs import DisparateImpactRemoverRS

__all__ += ["DisparateImpactRemoverRS"]
