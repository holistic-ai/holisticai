from __future__ import annotations

from typing import Union

from holisticai.utils.models.cluster._utils import distance
from numpy.random import RandomState


class KCenters:
    def __init__(self, n_clusters=5, random_state: Union[RandomState, int] = 42):
        """
        k (int) : Number of centers to be identified
        """
        self.k = n_clusters
        self.random_state = RandomState(random_state)

    def fit(self, data):
        """
        Performs the k-centers algorithm.

        Args:
                data (list) : Points in the dataset
        """

        self.data = data
        self.centers = [int(self.random_state.randint(0, len(self.data), 1))]
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

        Returns:
                mapping (list) : tuples of the form (point, center)
        """
        return [
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
