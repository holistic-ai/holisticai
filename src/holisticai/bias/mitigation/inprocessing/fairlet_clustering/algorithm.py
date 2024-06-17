import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin


class FairletClusteringAlgorithm:
    def __init__(self, decomposition, clustering_model):
        self.decomposition = decomposition
        self.clustering_model = clustering_model

    def fit(self, X, group_a, group_b, decompose=None):
        if decompose is not None:
            fairlets, fairlet_centers, fairlet_costs = decompose
        else:
            fairlets, fairlet_centers, fairlet_costs = self.decomposition.fit_transform(X, group_a, group_b)

        self.clustering_model.fit([X[i] for i in fairlet_centers])
        mapping = self.clustering_model.assign()

        self.labels = np.zeros(len(X), dtype="int32")
        for fairlet_id, final_cluster in mapping:
            self.labels[fairlets[fairlet_id]] = int(fairlet_centers[final_cluster])
        self.centers = np.array([fairlet_centers[i] for i in self.clustering_model.centers])
        self.cluster_centers_ = X[self.centers]
        self.cost = np.max(np.min(pairwise_distances(X, Y=X[self.centers]), axis=1))
        self.X = X

    def predict(self, X):
        """
        Decsription
        -----------
        Assigning every point in the dataset to the closest center.

        Paramters
        ---------
            X: matrix-like
            input dataset.

        Returns
        -------
            mapping : list of centers
        """
        fairlets_midxs = pairwise_distances_argmin(X, Y=self.X)
        return self.labels[fairlets_midxs]
