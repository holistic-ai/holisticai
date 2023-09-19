from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator

from holisticai.utils.transformers.bias import BMInprocessing as BMImp
from holisticai.utils.transformers.bias import SensitiveGroups

from .algorithm import KMediamClusteringAlgorithm


class FairKmedianClustering(BaseEstimator, BMImp):
    """
    Fair K-median clustering inprocessing bias mitigation is an approximation algorithms for
    group representative k-median clustering. Implementation of Algorithm 2. from [1]. The
    algorithm reduces the max group cost function.

    Reference:
    ----------
        [1] Abbasi, Mohsen, Aditya Bhaskara, and Suresh Venkatasubramanian. "Fair clustering via
        equitable group representations." Proceedings of the 2021 ACM Conference on Fairness,
        Accountability, and Transparency. 2021.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        max_iter: int = 1000,
        seed: int = None,
        strategy: Optional[str] = "LS",
        verbose: Optional[int] = 0,
    ):
        """
        Parameters
        ----------

        n_clusters: int
            number of clusters.

        max_iter: int
            Max number of iteration for LS or epochs for GA.

        strategy: str
            Minimization method used. Available:
            - LS (Local Search).
            - GA (Genetic Algorithm)

        verbose: int
            if > 0 , print progress information.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self.seed = seed
        self.strategy = strategy
        self.sensgroup = SensitiveGroups()
        self.algorithm = KMediamClusteringAlgorithm(
            n_clusters=n_clusters, max_iter=max_iter, strategy=strategy, verbose=verbose
        )

    def fit(self, X, group_a, group_b):
        """
        Fit model using Fair K-median Clustering.

        Parameters
        ----------
        X : matrix-like
            Input matrix
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        -------
            Self
        """
        np.random.seed(self.seed)

        params = self._load_data(X=X, group_a=group_a, group_b=group_b)
        X = params["X"]
        group_a = params["group_a"]
        group_b = params["group_b"]

        sensitive_groups = np.c_[group_a, group_b]
        p_attr = np.array(
            self.sensgroup.fit_transform(sensitive_groups, convert_numeric=True)
        )

        self.algorithm.fit(X, p_attr)
        return self

    @property
    def labels_(self):
        return self.algorithm.labels_

    @property
    def cluster_centers_(self):
        return self.algorithm.cluster_centers_
