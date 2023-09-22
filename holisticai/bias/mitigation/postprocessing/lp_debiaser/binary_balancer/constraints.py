from abc import abstractmethod
from itertools import combinations

import numpy as np


class Losses:
    def __call__(self, dr, overall_rates):
        # Getting the overall error rates and group proportions
        s = overall_rates.acc
        e = 1 - s

        # Setting up the coefficients for the objective function
        obj_coefs = np.array([[(s - e) * r[0], (e - s) * r[1]] for r in dr]).flatten()
        return obj_coefs


class ConstraintBase:
    @abstractmethod
    def get_conditions(self):
        pass

    def _get_rates(self, group_rates, groups):
        # Generating the pairs for comparison
        n_groups = len(groups)
        group_combos = list(combinations(groups, 2))
        id_combos = list(combinations(range(n_groups), 2))

        # Pair drop to keep things full-rank with 3 or more groups
        if n_groups > 2:
            n_comp = n_groups - 1
            group_combos = group_combos[:n_comp]
            id_combos = id_combos[:n_comp]

        col_combos = np.array(id_combos) * 2
        n_pairs = len(group_combos)

        # Making empty matrices to hold the pairwise constraint coefficients
        tprs = np.zeros(shape=(n_pairs, 2 * n_groups))
        fprs = np.zeros(shape=(n_pairs, 2 * n_groups))

        # Filling in the constraint matrices
        for i, cols in enumerate(col_combos):
            # Fetching the group-specific rates
            gc = group_combos[i]
            g0 = group_rates[gc[0]]
            g1 = group_rates[gc[1]]

            # Filling in the group-specific coefficients
            tprs[i, cols[0]] = g0.fnr
            tprs[i, cols[0] + 1] = g0.tpr
            tprs[i, cols[1]] = -g1.fnr
            tprs[i, cols[1] + 1] = -g1.tpr

            fprs[i, cols[0]] = g0.tnr
            fprs[i, cols[0] + 1] = g0.fpr
            fprs[i, cols[1]] = -g1.tnr
            fprs[i, cols[1] + 1] = -g1.fpr

        # Choosing whether to go with equalized odds or opportunity
        return tprs, fprs


class EqualizedOdds(ConstraintBase):
    def get_conditions(self, group_rates, groups):
        tprs, fprs = self._get_rates(group_rates, groups)
        con = np.vstack((tprs, fprs))
        con_bounds = np.zeros(con.shape[0])
        return con, con_bounds


class EqualizedOpportunity(ConstraintBase):
    def get_conditions(self, group_rates, groups):
        tprs, fprs = self._get_rates(group_rates, groups)
        con = tprs
        con_bounds = np.zeros(con.shape[0])
        return con, con_bounds
