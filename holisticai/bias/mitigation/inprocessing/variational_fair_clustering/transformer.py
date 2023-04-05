from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator

from holisticai.utils.transformers.bias import BMInprocessing as BMImp
from holisticai.utils.transformers.bias import SensitiveGroups

from .algorithm import FairClusteringAlgorithm


class VariationalFairClustering(BaseEstimator, BMImp):
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
        n_clusters: Optional[int],
        lipchitz_value: Optional[str] = 1,
        lmbda: Optional[float] = 0.7,
        method: Optional[str] = "kmeans",
        normalize_input: Optional[bool] = True,
        seed: Optional[int] = None,
        verbose: Optional[int] = 0,
    ):
        """
        Parameters
        ----------
            n_clusters : int
                The number of clusters to form as well as the number of centroids to generate.

            lipchitz_value : float
                Lipchitz value in bound update

            lmbda : float
                specified lambda parameter

            method : str
                cluster option : {'kmeans', 'kmedian'} (TODO: 'ncut' take too much time consuming)

            normalize_input : str
                Normalize input data X

            seed : int
                Random seed.

            verbose : bool
                If true , print metrics
        """
        # Constant parameters
        self.algorithm = FairClusteringAlgorithm(
            K=n_clusters,
            L=lipchitz_value,
            lmbda=lmbda,
            method=method,
            normalize_input=normalize_input,
            verbose=verbose,
        )
        self.seed = seed
        self.n_clusters = n_clusters
        self.lipchitz_value = lipchitz_value
        self.lmbda = lmbda
        self.method = method
        self.normalize_input = normalize_input
        self.verbose = verbose
        self.sens_group = SensitiveGroups()

    def fit(
        self,
        X: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Fit the model

        Description
        -----------
        Learn a fair cluster.

        Parameters
        ----------

        X : numpy array
            input matrix

        group_a : numpy array
            binary mask vector

        group_b : numpy array
            binary mask vector

        Returns
        -------
        the same object
        """
        params = self._load_data(X=X, group_a=group_a, group_b=group_b)
        X = params["X"]
        group_a = params["group_a"]
        group_b = params["group_b"]
        p_attr = self.sens_group.fit_transform(
            np.c_[group_a, group_b], convert_numeric=True
        )
        self.algorithm.fit(
            X=X, p_attr=p_attr, random_state=np.random.RandomState(self.seed)
        )
        return self

    @property
    def cluster_centers_(self):
        return self.algorithm.fair_clustering.C

    @property
    def labels_(self):
        return self.algorithm.fair_clustering.l

    def predict(self, X: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Prediction

        Description
        ----------
        Predict cluster for the given samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy array
            Test samples.

        group_a : numpy array
            binary mask vector

        group_b : numpy array
            binary mask vector


        Returns
        -------

        numpy.ndarray: Predicted output per sample.
        """
        params = self._load_data(X=X, group_a=group_a, group_b=group_b)
        X = params["X"]
        group_a = params["group_a"]
        group_b = params["group_b"]
        p_attr = self.sens_group.transform(
            np.c_[group_a, group_b], convert_numeric=True
        )
        return self.algorithm.predict(X, p_attr)

    def fit_predict(self, X: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Prediction

        Description
        ----------
        Fit and Predict the cluster for the given samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy array
            Test samples.

        group_a : numpy array
            binary mask vector

        group_b : numpy array
            binary mask vector


        Returns
        -------

        numpy.ndarray: Predicted cluster per sample.
        """
        self.fit(X, group_a, group_b)
        return self.labels_
