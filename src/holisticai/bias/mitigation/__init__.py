"""
The :mod:`holisticai.bias.mitigation` module includes preprocessing, inprocessing and postprocessing bias mitigation algorithms.
"""

# inprocessing algorithm classes
from holisticai.bias.mitigation.inprocessing import (
    AdversarialDebiasing,
    BlindSpotAwareMF,
    DebiasingLearningMF,
    ExponentiatedGradientReduction,
    FairKCenterClustering,
    FairKMedianClustering,
    FairletClustering,
    FairRec,
    GridSearchReduction,
    MetaFairClassifier,
    PopularityPropensityMF,
    PrejudiceRemover,
    VariationalFairClustering,
)

# postprocessing algorithm classes
from holisticai.bias.mitigation.postprocessing import (
    MCMF,
    CalibratedEqualizedOdds,
    DebiasingExposure,
    EqualizedOdds,
    FairTopK,
    LPDebiaserBinary,
    LPDebiaserMulticlass,
    MLDebiaser,
    PluginEstimationAndCalibration,
    RejectOptionClassification,
    WassersteinBarycenter,
)

# preprocessing algorithm classes
from holisticai.bias.mitigation.preprocessing import (
    CorrelationRemover,
    FairletClusteringPreprocessing,
    LearningFairRepresentation,
    Reweighing,
)

# all
__all__ = [
    "AdversarialDebiasing",
    "CorrelationRemover",
    "Reweighing",
    "LearningFairRepresentation",
    "ExponentiatedGradientReduction",
    "GridSearchReduction",
    "CalibratedEqualizedOdds",
    "EqualizedOdds",
    "RejectOptionClassification",
    "WassersteinBarycenter",
    "MLDebiaser",
    "LPDebiaserBinary",
    "LPDebiaserMulticlass",
    "PluginEstimationAndCalibration",
    "PrejudiceRemover",
    "MetaFairClassifier",
    "VariationalFairClustering",
    "FairletClustering",
    "FairletClusteringPreprocessing",
    "FairKCenterClustering",
    "FairKMedianClustering",
    "BlindSpotAwareMF",
    "DebiasingLearningMF",
    "PopularityPropensityMF",
    "FairRec",
    "DebiasingExposure",
    "FairTopK",
    "MCMF",
]

import importlib
from typing import Literal

networkx_spec = importlib.util.find_spec("networkx")
if networkx_spec is not None:
    from holisticai.bias.mitigation.postprocessing import DisparateImpactRemoverRS
    from holisticai.bias.mitigation.preprocessing import DisparateImpactRemover

__all__ += ["DisparateImpactRemoverRS", "DisparateImpactRemover"]

cvxpy_spec = importlib.util.find_spec("cvxpy")
if cvxpy_spec is not None:
    from holisticai.bias.mitigation.inprocessing.fair_scoring_classifier.transformer import FairScoreClassifier
__all__ += ["FairScoreClassifier"]

MITIGATOR_NAME = Literal[
    "CorrelationRemover",
    "Reweighing",
    "LearningFairRepresentation",
    "ExponentiatedGradientReduction",
    "GridSearchReduction",
    "CalibratedEqualizedOdds",
    "EqualizedOdds",
    "RejectOptionClassification",
    "WassersteinBarycenter",
    "MLDebiaser",
    "LPDebiaserBinary",
    "LPDebiaserMulticlass",
    "PluginEstimationAndCalibration",
    "PrejudiceRemover",
    "MetaFairClassifier",
    "VariationalFairClustering",
    "FairletClustering",
    "FairletClusteringPreprocessing",
    "FairKCenterClustering",
    "FairKMedianClustering",
    "BlindSpotAwareMF",
    "DebiasingLearningMF",
    "PopularityPropensityMF",
    "FairRec",
    "FairScoreClassifier",
    "DebiasingExposure",
    "FairTopK",
    "MCMF",
    "AdversarialDebiasing",
    "DisparateImpactRemoverRS",
    "DisparateImpactRemover",
]
