# imports
from .exponentiated_gradient.transformer import ExponentiatedGradientReduction
from .fair_k_center_clustering.transformer import FairKCenterClustering
from .fair_k_mediam_clustering.transformer import FairKMedianClustering
from .fairlet_clustering.transformer import FairletClustering
from .grid_search.transformer import GridSearchReduction
from .matrix_factorization.blind_spot_aware import BlindSpotAwareMF
from .matrix_factorization.debiasing_learning.transformer import DebiasingLearningMF
from .matrix_factorization.popularity_propensity import PopularityPropensityMF
from .meta_fair_classifier.transformer import MetaFairClassifier
from .prejudice_remover.transformer import PrejudiceRemover
from .two_sided_fairness.transformer import FairRec
from .variational_fair_clustering.transformer import VariationalFairClustering

__all__ = [
    "ExponentiatedGradientReduction",
    "GridSearchReduction",
    "PrejudiceRemover",
    "MetaFairClassifier",
    "VariationalFairClustering",
    "FairKCenterClustering",
    "FairKMedianClustering",
    "FairletClustering",
    "BlindSpotAwareMF",
    "DebiasingLearningMF",
    "PopularityPropensityMF",
    "FairRec",
]
import importlib.util

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    from .adversarial_debiasing.torch.transformer import AdversarialDebiasing
    __all__.append("AdversarialDebiasing")

cvxpy_spec = importlib.util.find_spec("cvxpy")
if cvxpy_spec is not None:
    from .fair_scoring_classifier.transformer import FairScoreClassifier
    __all__.append("FairScoreClassifier")
