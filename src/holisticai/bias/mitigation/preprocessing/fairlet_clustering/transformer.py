from __future__ import annotations

from typing import Optional, Union

import numpy as np
from holisticai.bias.mitigation.commons.fairlet_clustering.decompositions import (
    DecompositionMixin,
    ScalableFairletDecomposition,
    VanillaFairletDecomposition,
)
from holisticai.utils.models.cluster import KCenters, KMedoids
from holisticai.utils.transformers.bias import BMPreprocessing as BMPre
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_distances_argmin

DECOMPOSITION_CATALOG = {
    "Scalable": ScalableFairletDecomposition,
    "Vanilla": VanillaFairletDecomposition,
}
CLUSTERING_CATALOG = {"KCenter": KCenters, "KMedoids": KMedoids}


class FairletClusteringPreprocessing(BaseEstimator, BMPre):
    """
    Fairlet decomposition [1]_ is a pre-processing approach that computes\
    fair micro-clusters where fairness is guaranteed. They then use\
    the fairlet centers as a newly transformed dataset from the original.\
    This transformed fairlet-based dataset is then provided to vanilla\
    clustering algorithms, and hence, we obtain approximately\
    fair clustering outputs as a result of the fairlets themselves being fair.

    Parameters
    ----------
    decomposition : str, optional
        Fairlet decomposition strategy, available: Vanilla, Scalable, MCF. Default is Vanilla.

    p : int, optional
        fairlet decomposition parameter for Vanilla and Scalable strategy. Default is 1.

    q : int, optional
        fairlet decomposition parameter for Vanilla and Scalable strategy. Default is 3.

    seed : int, optional
        Random seed. Default is None.

    References
    ----------
    .. [1] `Backurs, Arturs, et al. "Scalable fair clustering." International Conference on
        Machine Learning. PMLR, 2019.`
    """

    def __init__(
        self,
        decomposition: Union[str, DecompositionMixin] = "Vanilla",
        p: Optional[str] = 1,
        q: Optional[float] = 3,
        seed: Optional[int] = None,
    ):
        self.decomposition = DECOMPOSITION_CATALOG[decomposition](p=p, q=q)
        self.p = p
        self.q = q
        self.seed = seed

    def fit_transform(
        self,
        X: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Fits the model by learning a fair cluster.

        Parameters
        ----------

        X : matrix-like
            input matrix

        group_a : array-like
            binary mask vector

        group_b : array-like
            binary mask vector

        sample_weight : array-like, optional
            Samples weights vector. Default is None.

        Returns
        -------
        matrix
            Transformed matrix
        """
        params = self._load_data(X=X, sample_weight=sample_weight, group_a=group_a, group_b=group_b)
        x = params["X"]
        sample_weight = params["sample_weight"]
        group_a = params["group_a"].astype("int32")
        group_b = params["group_b"].astype("int32")
        np.random.seed(self.seed)
        fairlets, fairlet_centers, fairlet_costs = self.decomposition.fit_transform(x, group_a, group_b)
        xt = np.zeros_like(x)
        mapping = np.zeros(len(x), dtype="int32")
        centers = np.array([x[fairlet_center] for fairlet_center in fairlet_centers])
        for i, fairlet in enumerate(fairlets):
            xt[fairlet] = x[fairlet_centers[i]]
            mapping[fairlet] = i
            sample_weight[fairlet] = len(fairlet) / len(x)

        self._update_estimator_param("sample_weight", sample_weight)
        self.sample_weight = sample_weight
        self.X = x
        self.mapping = mapping
        self.centers = centers
        return xt

    def transform(self, X):
        """
        Transforms the model by learning a fair cluster.

        Parameters
        ----------
        X : matrix-like
            input matrix

        Returns
        -------
        matrix
            Transformed matrix
        """
        fairlets_midxs = pairwise_distances_argmin(X, Y=self.X)
        return self.centers[self.mapping[fairlets_midxs]]
