from abc import abstractmethod
from itertools import combinations

import numpy as np


class MacroLosses:
    def __call__(self, cp_mats):
        off_loss = [
            [np.delete(a, i, 0).sum(0) for i in range(self.n_classes)] for a in cp_mats
        ]
        obj = np.array(off_loss).flatten()
        return obj


class MicroLosses:
    def __call__(self, cp_mats):
        u = np.array(
            [[np.delete(a, i, 0) for i in range(self.n_classes)] for a in cp_mats]
        )
        p = np.array(
            [
                [np.delete(a, i).reshape(-1, 1) for i in range(self.n_classes)]
                for a in self.p_vecs
            ]
        )
        w = np.array(
            [
                [p[i, j] * u[i, j] for j in range(self.n_classes)]
                for i in range(self.n_groups)
            ]
        )
        obj = w.sum(2).flatten()
        return obj


class ConstraintBase:
    @abstractmethod
    def get_conditions(self):
        pass

    def _constraint_weights(self, p_vecs, p_a, cp_mats):
        # Getting the costraint weights
        constraints_by_group = [
            self.__get_constraints(p_vecs[i], p_a[i], cp_mats[i])
            for i in range(self.n_groups)
        ]

        # Arranging the constraint weights by group comparisons
        return self.__pair_constraints(constraints_by_group, cp_mats)

    def __get_constraints(self, p_vec, p_a, cp_mat):
        """Calculates TPR and FPR weights for the constraint matrix"""
        # Shortening the vars to keep things clean
        p = p_vec
        M = cp_mat

        # Setting up the matrix of parameter weights
        n_classes = M.shape[0]
        n_params = n_classes**2
        tpr = np.zeros(shape=(n_classes, n_params))
        fpr = np.zeros(shape=(n_classes, n_params))

        # Filling in the weights
        for i in range(n_classes):
            # Dropping row to calculate FPR
            p_i = np.delete(p, i)
            M_i = np.delete(M, i, 0)

            start = i * (n_classes)
            end = start + n_classes
            fpr[i, start:end] = np.dot(p_i, M_i) / p_i.sum()
            tpr[i, start:end] = M[i]

        # Reshaping the off-diagonal constraints
        strict = np.zeros(shape=(n_params, n_params))
        # A = np.array(p.T / p_a).T
        # B = np.array(M.T * A).T
        B = M
        for i in range(n_classes):
            start = i * n_classes
            end = start + n_classes
            strict[start:end, start:end] = B

        return tpr, fpr, strict

    def __pair_constraints(self, constraints, cp_mats):
        """Takes the output of constraint_weights() and returns a matrix
        of the pairwise constraints
        """
        # Setting up the preliminaries
        tprs = np.array([c[0] for c in constraints])
        fprs = np.array([c[1] for c in constraints])
        strict = np.array([c[2] for c in constraints])
        n_params = tprs.shape[2]
        n_classes = tprs.shape[1]
        n_groups = self.n_groups
        if n_groups > 2:
            group_combos = list(combinations(range(n_groups), 2))[:-1]
        else:
            group_combos = [(0, 1)]

        n_pairs = len(group_combos)

        # Setting up the empty matrices
        tpr_cons = np.zeros(shape=(n_pairs, n_groups, n_classes, n_params))
        fpr_cons = np.zeros(shape=(n_pairs, n_groups, n_classes, n_params))
        strict_cons = np.zeros(shape=(n_pairs, n_groups, n_params, n_params))

        # Filling in the constraint comparisons
        for i, c in enumerate(group_combos):
            # Getting the original diffs for flipping
            diffs = cp_mats[c[0]] - cp_mats[c[1]]
            tpr_flip = np.sign(np.diag(diffs)).reshape(-1, 1)
            fpr_flip = np.sign(np.sum(fprs[c[0]], 1) - np.sum(fprs[c[1]], 1))
            fpr_flip = fpr_flip.reshape(-1, 1)
            fpr_flip[np.where(fpr_flip == 0)] = 1

            # Filling in the constraints
            tpr_cons[i, c[0]] = tpr_flip * tprs[c[0]]
            tpr_cons[i, c[1]] = tpr_flip * -1 * tprs[c[1]]
            fpr_cons[i, c[0]] = fpr_flip * fprs[c[0]]
            fpr_cons[i, c[1]] = fpr_flip * -1 * fprs[c[1]]
            strict_cons[i, c[0]] = strict[c[0]]
            strict_cons[i, c[1]] = -1 * strict[c[1]]

        # Filling in the norm constraints
        one_cons = np.zeros(shape=(n_groups * n_classes, n_groups * n_params))
        cols = np.array(list(range(0, n_groups * n_classes**2, n_classes)))
        cols = cols.reshape(n_groups, n_classes)
        i = 0
        for c in cols:
            for j in range(n_classes):
                one_cons[i, c + j] = 1
                i += 1

        # Reshaping the arrays
        tpr_cons = np.concatenate([np.hstack(m) for m in tpr_cons])
        fpr_cons = np.concatenate([np.hstack(m) for m in fpr_cons])
        strict_cons = np.concatenate([np.hstack(m) for m in strict_cons])

        return tpr_cons, fpr_cons, strict_cons, one_cons


class EqualizedOdds(ConstraintBase):
    def get_conditions(self, p_vecs, p_a, cp_mats):
        tpr_cons, fpr_cons, strict_cons, norm_cons = self._constraint_weights(
            p_vecs, p_a, cp_mats
        )
        norm_bounds = np.repeat(1, norm_cons.shape[0])
        con = np.concatenate([tpr_cons, fpr_cons, norm_cons])
        eo_bounds = np.repeat(0, tpr_cons.shape[0] * 2)
        con_bounds = np.concatenate([eo_bounds, norm_bounds])
        return con, con_bounds


class EqualizedOpportunity(ConstraintBase):
    def get_conditions(self, p_vecs, p_a, cp_mats):
        tpr_cons, fpr_cons, strict_cons, norm_cons = self._constraint_weights(
            p_vecs, p_a, cp_mats
        )
        norm_bounds = np.repeat(1, norm_cons.shape[0])
        con = np.concatenate([tpr_cons, norm_cons])
        tpr_bounds = np.repeat(0, tpr_cons.shape[0])
        con_bounds = np.concatenate([tpr_bounds, norm_bounds])
        return con, con_bounds


class Strict(ConstraintBase):
    def get_conditions(self, p_vecs, p_a, cp_mats):
        tpr_cons, fpr_cons, strict_cons, norm_cons = self._constraint_weights(
            p_vecs, p_a, cp_mats
        )
        norm_bounds = np.repeat(1, norm_cons.shape[0])
        con = np.concatenate([strict_cons, norm_cons])
        strict_bounds = np.repeat(0, strict_cons.shape[0])
        con_bounds = np.concatenate([strict_bounds, norm_bounds])
        return con, con_bounds
