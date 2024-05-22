import numpy as np

from ._base import DecompositionMixin


class VanillaFairletDecomposition(DecompositionMixin):
    def __init__(self, p, q):
        self.fairlets = []
        self.fairlet_centers = []
        self.p = p
        self.q = q

    def balanced(self, p, q, r, b):
        if r == 0 and b == 0:
            return True
        if r == 0 or b == 0:
            return False
        return min(r * 1.0 / b, b * 1.0 / r) >= p * 1.0 / q

    def make_fairlet(self, points, dataset):
        "Adds fairlet to fairlet decomposition, returns median cost"
        self.fairlets.append(points)
        cost_list = [
            sum([np.linalg.norm(dataset[center] - dataset[point]) for point in points])
            for center in points
        ]
        cost, center = min((cost, center) for (center, cost) in enumerate(cost_list))
        self.fairlet_centers.append(points[center])
        return cost

    def fit_transform(self, dataset, group_a, group_b):
        """
        Computes vanilla (p,q)-fairlet decomposition of given points (Lemma 3 in NIPS17 paper).
        Returns cost.
        Input: Balance parameters p,q which are non-negative integers satisfying p<=q and gcd(p,q)=1.
        "blues" and "reds" are sets of points indices with balance at least p/q.
        """
        blues = list(np.where(group_a)[0])
        reds = list(np.where(group_b)[0])
        return self.fairlets, self.fairlet_centers, self.decompose(blues, reds, dataset)

    def decompose(self, blues, reds, dataset):
        p = self.p
        q = self.q
        assert p <= q, "Please use balance parameters in the correct order"
        if len(reds) < len(blues):
            temp = blues
            blues = reds
            reds = temp
        R = len(reds)
        B = len(blues)
        assert self.balanced(p, q, R, B), (
            "Input sets are unbalanced: " + str(R) + "," + str(B)
        )

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
            cost += self.make_fairlet(
                reds[r0 : r0 + (R - r0) - (B - b0) + p] + blues[b0 : b0 + p], dataset
            )
            r0 += (R - r0) - (B - b0) + p
            b0 += p
        assert R - r0 == B - b0, "Error in computing fairlet decomposition"
        for i in range(R - r0):
            cost += self.make_fairlet([reds[r0 + i], blues[b0 + i]], dataset)
        return cost
