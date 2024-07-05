import numpy as np
from holisticai.bias.mitigation.commons.fairlet_clustering._utils import distance


class KCenters:
    """
    K-Centers algorithm for clustering.

    Parameters
    ----------
    n_clusters : int, optional
        The number of clusters to identify. Default is 5.

    Attributes
    ----------
    k : int
        The number of centers to identify.
    data : list
        The points in the dataset.
    centers : list
        The indices of the centers.
    costs : list
        The cost of the centers.
    cluster_centers_ : list
        The centers of the clusters.
    labels : list
        The mapping of points to their closest centers.

    Methods
    -------
    fit(data)
        Performs the k-centers algorithm.
    assign()
        Assigns every point in the dataset to the closest center.
    """

    def __init__(self, n_clusters=5):
        self.k = n_clusters

    def fit(self, data):
        """
        Performs the k-centers algorithm.

        Parameters
        ----------
        data : list
            The points in the dataset.

        Returns
        -------
        None
        """
        # Randomly choosing an initial center
        # random.seed(42)

        self.data = data
        self.centers = [int(np.random.randint(0, len(self.data), 1))]
        self.costs = []

        while True:
            # Remaining points in the data set
            rem_points = list(set(range(len(self.data))) - set(self.centers))

            # Finding the point which has the closest center most far-off
            point_center = [(i, min([distance(self.data[i], self.data[j]) for j in self.centers])) for i in rem_points]

            point_center = sorted(point_center, key=lambda x: x[1], reverse=True)

            self.costs.append(point_center[0][1])
            if len(self.centers) < self.k:
                self.centers.append(point_center[0][0])
            else:
                break

        self.cluster_centers_ = [self.data[j] for j in self.centers]
        self.labels = self.assign()

    def assign(self):
        """
        Assigning every point in the dataset to the closest center.

        Returns
        -------
        list
            The mapping of points to their closest centers.
        """
        mapping = [
            (
                i,
                sorted(
                    [(j, distance(self.data[i], self.data[j])) for j in self.centers],
                    key=lambda x: x[1],
                    reverse=False,
                )[0][0],
            )
            for i in range(len(self.data))
        ]

        return mapping
