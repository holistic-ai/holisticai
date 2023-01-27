from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin

from holisticai.utils.transformers.bias import BMInprocessing as BMImp
from holisticai.utils.transformers.bias import SensitiveGroups

from .algorithms import (
    fair_k_center_APPROX,
    heuristic_greedy_on_each_group,
    heuristic_greedy_till_constraint_is_satisfied,
)

STRATEGIES_CATALOG = {
    "Fair K-Center": fair_k_center_APPROX,
    "Heuristic Greedy by Group": heuristic_greedy_on_each_group,
    "Heuristic Greedy by Constraint": heuristic_greedy_till_constraint_is_satisfied,
}


class FairKCenterClustering(BaseEstimator, BMImp):
    """
    Fair K-Center Clustering inprocessing bias mitigation implements an approximation algorithm
    for the k-centers problem under the fairness contraint with running time linear in the
    size of the dataset and k (number of cluster).

    Reference
    ---------
        Kleindessner, Matthäus, Pranjal Awasthi, and Jamie Morgenstern. "Fair k-center clustering
        for data summarization." International Conference on Machine Learning. PMLR, 2019.
    """

    def __init__(
        self,
        req_nr_per_group: Optional[list] = [200, 200],
        nr_initially_given: Optional[int] = 100,
        strategy: Optional[str] = "Fair K-Center",
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        req_nr_per_group : list
            Number of cluster for each group that will be founded.
            - Integer-vector of length m with entries in 0,...,k.
            - Sum of all entries must be equal to k (total number of clusters).

        nr_initially_given: int
            Number of initial random centers.

        strategy: Strategy used to compute the cluster centers. Available are:
            - 'Fair K-Center' (default)
            - 'Heuristic Greedy by Group'
            - 'Heuristic Greedy by Constraint'

        seed: int,
            Initial random seed.
        """
        self.req_nr_per_group = np.array(req_nr_per_group)
        self.nr_initially_given = nr_initially_given
        self.strategy = strategy
        self.seed = seed
        self.sensgroup = SensitiveGroups()

    def fit(self, X, group_a, group_b):
        """
        Fit model using Fair K-Center Clustering.

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

        n = len(X)
        dmat = pairwise_distances(X, metric="l1")
        initially_given = np.random.choice(
            n, size=self.nr_initially_given, replace=False
        )
        centers = STRATEGIES_CATALOG[self.strategy](
            dmat, p_attr, self.req_nr_per_group, initially_given
        )
        cost = np.amax(
            np.amin(
                dmat[np.ix_(np.hstack((centers, initially_given)), np.arange(n))],
                axis=0,
            )
        )
        self.centers = centers
        self.initially_given = initially_given
        self.cost = cost

        self.centroids = X[self.centers]
        self.all_centroids = X[self.all_centers]

        self.labels_ = self.all_centers[
            pairwise_distances_argmin(X, Y=self.all_centroids, metric="l1")
        ]
        self.center_groups_ = p_attr[self.all_centers]

    @property
    def all_centers(self):
        return np.concatenate([self.centers, self.initially_given], axis=0)
