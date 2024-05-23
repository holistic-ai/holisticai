"""
The :mod:`holisticai.bias.mitigation` module includes preprocessing, inprocessing and postprocessing bias mitigation algorithms.
"""

# inprocessing algorithm classes
from holisticai.mitigation.bias.inprocessing import (
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
from holisticai.mitigation.bias.postprocessing import (
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
from holisticai.mitigation.bias.preprocessing import (
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
    "MCMF",
]

import importlib

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    from holisticai.mitigation.bias.inprocessing import AdversarialDebiasing
__all__ += ["AdversarialDebiasing"]

networkx_spec = importlib.util.find_spec("networkx")
if networkx_spec is not None:
    from holisticai.mitigation.bias.postprocessing import DisparateImpactRemoverRS
    from holisticai.mitigation.bias.preprocessing import DisparateImpactRemover

__all__ += ["DisparateImpactRemoverRS", "DisparateImpactRemover"]
