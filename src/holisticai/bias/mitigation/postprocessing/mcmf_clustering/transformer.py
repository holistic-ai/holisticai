from __future__ import annotations

from typing import TYPE_CHECKING

from holisticai.bias.mitigation.postprocessing.mcmf_clustering.algorithm import Algorithm
from holisticai.utils.transformers.bias import BMPostprocessing as BMPost

if TYPE_CHECKING:
    import numpy as np


class MCMF(BMPost):
    """
    Minimal Cluster Modification for Fairnes (MCMF) [1]_ is focused on the minimal change it so that the clustering is still\
    of good quality and fairer.

    Parameters
    ----------
    metric : str
        Measure function used in the objective function.
        The metrics available are:
        ["constant", "L1", "L2"]

    solver : str
        Algorithm name used to solve the standard form problem. Solver supported must depend of your scipy package version.
        for scipy 1.9.0 the solvers available are:
        ["highs", "highs-ds", "highs-ipm"]

    group_mode : str
        Set what groups will be fitted: ['a', 'b', 'ab']
    verbose : int
        If > 0 , then print logs.

    References
    ----------
        .. [1] Davidson, Ian, and S. S. Ravi. "Making existing clusterings fairer: Algorithms, complexity results and insights."\
        Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 04. 2020.
    """

    def __init__(self, metric: str = "L1", solver: str = "highs", group_mode="a", verbose=0):
        self.metric = metric
        self.group_mode = group_mode
        self.solver = solver
        self.verbose = verbose
        self.algorithm = Algorithm(metric=metric, solver=solver, verbose=verbose)

    def fit(self):
        """
        Fit the MCMF algorithm.

        Returns
        -------
        self
        """
        return self

    def fit_transform(
        self,
        X: np.ndarray,
        y_pred: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
        centroids: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Fit and transform the MCMF algorithm.

        Description
        ----------
        Fit the MCMF algorithm and transform the predicted vector.

        Parameters
        ----------
        X : matrix-like
            Input matrix (nb_examples, nb_features)
        y_pred : array-like
            Predicted vector (nb_examples,)
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        centroids : ndarray
            Centroids array

        Returns
        -------
        dict
            A dictionary with new predictions
        """
        return self.transform(X, y_pred, group_a, group_b, centroids)

    def transform(
        self,
        X: np.ndarray,
        y_pred: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
        centroids: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Modify y_pred using MCMF algorithm.

        Description
        ----------
        Build parameters for the objetive function and call the solver to find the algorithm parameters.

        Parameters
        ----------
        X : matrix-like
            Input matrix (nb_examples, nb_features)
        y_pred : array-like
            Predicted vector (nb_examples,)
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        centroids : ndarray
            Centroids array

        Returns
        -------
        dict
            A dictionary with new predictions
        """
        params = self._load_data(X=X, y_pred=y_pred, group_a=group_a, group_b=group_b)

        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1
        X = params["X"]
        y_pred = params["y_pred"]

        if isinstance(centroids, str):
            centroids = getattr(self.estimator_hdl.estimator, centroids)

        if self.group_mode in ["a", "b"]:
            group = group_a if self.group_mode == "a" else group_b
            new_y_pred = self.algorithm.transform(X=X, y_pred=y_pred, group=group, centroids=centroids)
        elif self.group_mode == "ab":
            new_y_pred = y_pred.copy()
            for group in [group_a, group_b]:
                new_y_pred = self.algorithm.transform(X=X, y_pred=new_y_pred, group=group, centroids=centroids)

        return {"y_pred": new_y_pred}
