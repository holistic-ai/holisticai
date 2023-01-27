"""
The :mod:`holisticai.bias.mitigation` module includes preprocessing, inprocessing and postprocessing bias mitigation algorithms.
"""

# inprocessing algorithm classes
from .inprocessing import (
    BlindSpotAwareMF,
    DebiasingLearningMF,
    ExponentiatedGradientReduction,
    FairKCenterClustering,
    FairKmedianClustering,
    FairletClustering,
    FairRec,
    FairScoreClassifier,
    GridSearchReduction,
    MetaFairClassifier,
    PopularityPropensityMF,
    PrejudiceRemover,
    VariationalFairClustering,
)

# postprocessing algorithm classes
from .postprocessing import (
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
from .preprocessing import (
    CorrelationRemover,
    FairletClusteringPreprocessing,
    LearningFairRepresentation,
    Reweighing,
)

# all
__all__ = [
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
    "FairKmedianClustering",
    "BlindSpotAwareMF",
    "DebiasingLearningMF",
    "PopularityPropensityMF",
    "FairRec",
    "FairScoreClassifier",
    "DebiasingExposure",
    "FairTopK",
]

import importlib

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    from .inprocessing import AdversarialDebiasing
__all__ += ["AdversarialDebiasing"]

networkx_spec = importlib.util.find_spec("networkx")
if networkx_spec is not None:
    from .postprocessing import DisparateImpactRemoverRS
    from .preprocessing import DisparateImpactRemover

__all__ += ["DisparateImpactRemoverRS", "DisparateImpactRemover"]
