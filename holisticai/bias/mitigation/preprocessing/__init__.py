# Imports
from .correlation_remover import CorrelationRemover
from .fairlet_clustering.transformer import FairletClusteringPreprocessing
from .learning_fair_representation import LearningFairRepresentation
from .reweighing import Reweighing

__all__ = [
    "LearningFairRepresentation",
    "Reweighing",
    "CorrelationRemover",
    "FairletClusteringPreprocessing",
]

import importlib

networkx_spec = importlib.util.find_spec("networkx")
if networkx_spec is not None:
    from .disparate_impact_remover import DisparateImpactRemover

__all__ += ["DisparateImpactRemover"]
