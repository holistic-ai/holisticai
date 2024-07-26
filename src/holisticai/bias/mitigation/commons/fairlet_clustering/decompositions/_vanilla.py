import numpy as np
from holisticai.bias.mitigation.commons.fairlet_clustering.decompositions._base import DecompositionMixin
from sklearn.metrics.pairwise import pairwise_distances


class VanillaFairletDecomposition(DecompositionMixin):
    """
    Vanilla (p,q)-fairlet decomposition of given points (Lemma 3 in NIPS17 paper).

    Parameters
    ----------
    p : int
        The balance parameter p.
    q : int
        The balance parameter q.

    Attributes
    ----------
    fairlets : list
        The list of fairlets.
    fairlet_centers : list
        The list of fairlet centers.
    p : int
        The balance parameter p.
    q : int
        The balance parameter q.

    Methods
    -------
    balanced(p, q, r, b)
        Checks if the input sets are balanced.
    make_fairlet(points, dataset)
        Adds fairlet to fairlet decomposition, returns median cost.
    fit_transform(dataset, group_a, group_b)
        Computes vanilla (p,q)-fairlet decomposition of given points.
    decompose(blues, reds, dataset)
        Decomposes the input sets into fairlets.
    """

    def __init__(self, p, q):
        self.fairlets = []
        self.fairlet_centers = []
        self.p = p
        self.q = q

    def balanced(self, p, q, r, b):
        """
        Checks if the input sets are balanced.

        Parameters
        ----------
        p : int
            The balance parameter p.
        q : int
            The balance parameter q.
        r : int
            The number of red points.
        b : int
            The number of blue points.

        Returns
        -------
        bool
            True if the input sets are balanced, False otherwise.
        """
        if r == 0 and b == 0:
            return True
        if r == 0 or b == 0:
            return False
        return min(r * 1.0 / b, b * 1.0 / r) >= p * 1.0 / q

    def make_fairlet(self, points, dataset):
        """
        Adds fairlet to fairlet decomposition, returns median cost

        Parameters
        ----------
        points : list
            The list of points.
        dataset : array-like
            The dataset.

        Returns
        -------
        float
            The median cost.
        """

        self.fairlets.append(points)
        cost_matrix = np.sum(pairwise_distances(dataset[points], Y=dataset[points]), axis=1)
        center = np.argmin(cost_matrix)
        cost = cost_matrix[center]
        self.fairlet_centers.append(points[center])
        return cost

    def fit_transform(self, dataset, group_a, group_b):
        """
        Computes vanilla (p,q)-fairlet decomposition of given points (Lemma 3 in NIPS17 paper).

        Parameters
        ----------
        dataset : array-like
            The dataset.
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        ------
        list
            The list of fairlets.
        list
            The list of fairlet centers.
        float
            The cost of the fairlet decomposition.
        """
        blues = list(np.where(group_a)[0])
        reds = list(np.where(group_b)[0])
        return (
            self.fairlets,
            self.fairlet_centers,
            self._decompose(blues, reds, dataset),
        )

    def _decompose(self, blues, reds, dataset):
        """
        Decomposes the input sets into fairlets.

        Parameters
        ----------
        blues : list
            The set of blue points.
        reds : list
            The set of red points.
        dataset : array-like
            The dataset.

        Returns
        -------
        float
            The cost of the fairlet decomposition.
        """
        p = self.p
        q = self.q
        assert p <= q, "Please use balance parameters in the correct order"
        if len(reds) < len(blues):
            temp = blues
            blues = reds
            reds = temp
        R = len(reds)
        B = len(blues)
        assert self.balanced(p, q, R, B), "Input sets are unbalanced: " + str(R) + "," + str(B)

        if R == 0 and B == 0:
            return 0

        b0 = 0
        r0 = 0
        cost = 0
        while (R - r0) - (B - b0) >= q - p and R - r0 >= q and B - b0 >= p:
            cost += self.make_fairlet(reds[r0 : r0 + q] + blues[b0 : b0 + p], dataset)
            r0 += q
            b0 += p
        if R - r0 + B - b0 >= 1 and R - r0 + B - b0 <= p + q:
            cost += self.make_fairlet(reds[r0:] + blues[b0:], dataset)
            r0 = R
            b0 = B
        elif R - r0 != B - b0 and B - b0 >= p:
            cost += self.make_fairlet(reds[r0 : r0 + (R - r0) - (B - b0) + p] + blues[b0 : b0 + p], dataset)
            r0 += (R - r0) - (B - b0) + p
            b0 += p
        assert R - r0 == B - b0, "Error in computing fairlet decomposition"
        for i in range(R - r0):
            cost += self.make_fairlet([reds[r0 + i], blues[b0 + i]], dataset)
        return cost
