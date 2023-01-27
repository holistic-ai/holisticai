from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_distances_argmin

from holisticai.bias.mitigation.commons.fairlet_clustering.decompositions import (
    DecompositionMixin,
    ScalableFairletDecomposition,
    VanillaFairletDecomposition,
)
from holisticai.utils.models.cluster import KCenters, KMedoids
from holisticai.utils.transformers.bias import BMPreprocessing as BMPre

DECOMPOSITION_CATALOG = {
    "Scalable": ScalableFairletDecomposition,
    "Vanilla": VanillaFairletDecomposition,
}
CLUSTERING_CATALOG = {"KCenter": KCenters, "KMedoids": KMedoids}


class FairletClusteringPreprocessing(BaseEstimator, BMPre):
    """
    Variational Fair Clustering helps you to find clusters with specified proportions
    of different demographic groups pertaining to a sensitive attribute of the dataset
    (group_a and group_b) for any well-known clustering method such as K-means, K-median
    or Spectral clustering (Normalized cut).


    References
    ----------
        Ziko, Imtiaz Masud, et al. "Variational fair clustering." Proceedings of the AAAI
        Conference on Artificial Intelligence. Vol. 35. No. 12. 2021.
    """

    def __init__(
        self,
        decomposition: Union["str", "DecompositionMixin"] = "Vanilla",
        p: Optional[str] = 1,
        q: Optional[float] = 3,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
            decomposition : str
                Fairlet decomposition strategy, available: Vanilla, Scalable, MCF

            p : int
                fairlet decomposition parameter for Vanilla and Scalable strategy

            q : int
                fairlet decomposition parameter for Vanilla and Scalable strategy

            seed : int
                Random seed.
        """
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
        Fit the model

        Description
        -----------
        Learn a fair cluster.

        Parameters
        ----------

        X : matrix-like
            input matrix

        group_a : array-like
            binary mask vector

        group_b : array-like
            binary mask vector

        sample_weight (optional) : array-like
            Samples weights vector

        Returns
        -------
        the same object
        """
        params = self._load_data(
            X=X, sample_weight=sample_weight, group_a=group_a, group_b=group_b
        )
        X = params["X"]
        sample_weight = params["sample_weight"]
        group_a = params["group_a"].astype("int32")
        group_b = params["group_b"].astype("int32")
        np.random.seed(self.seed)
        fairlets, fairlet_centers, fairlet_costs = self.decomposition.fit_transform(
            X, group_a, group_b
        )
        Xt = np.zeros_like(X)
        mapping = np.zeros(len(X), dtype="int32")
        centers = np.array([X[fairlet_center] for fairlet_center in fairlet_centers])
        for i, fairlet in enumerate(fairlets):
            Xt[fairlet] = X[fairlet_centers[i]]
            mapping[fairlet] = i
            sample_weight[fairlet] = len(fairlet) / len(X)

        self.update_estimator_param("sample_weight", sample_weight)
        self.sample_weight = sample_weight
        self.X = X
        self.mapping = mapping
        self.centers = centers
        return Xt

    def transform(self, X):
        fairlets_midxs = pairwise_distances_argmin(X, Y=self.X)
        return self.centers[self.mapping[fairlets_midxs]]
