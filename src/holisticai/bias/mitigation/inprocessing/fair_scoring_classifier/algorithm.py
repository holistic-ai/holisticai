from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from holisticai.bias.mitigation.inprocessing.fair_scoring_classifier.utils import get_class_count, get_class_indexes

logger = logging.getLogger(__name__)


class FairScoreClassifierAlgorithm:
    """
    Generates a classification model that integrates fairness constraints for multiclass classification. This algorithm
    returns a matrix of lambda coefficients that scores a given input vector. The higher the score, the higher the probability
    of the input vector to be classified as the majority class.

    References:
        Julien Rouzot, Julien Ferry, Marie-José Huguet. Learning Optimal Fair Scoring Systems for Multi-
        Class Classification. ICTAI 2022 - The 34th IEEE International Conference on Tools with Artificial
        Intelligence, Oct 2022, Virtual, United States.
    """

    def __init__(
        self,
        objectives: str,
        fairness_groups: list,
        fairness_labels: list,
        constraints: Optional[dict] = None,
        lambda_bound: int = 9,
        time_limit: int = 100,
        verbose: int = 0,
    ):
        """
        Init FairScoreClassifier object

        Parameters
        ----------
        objectives : str
            The bjectives list to be optimized.

        fairness_groups : list
            The sensitive groups indexes.

        fairness_labels : list
            The senstive labels indexes.

        constraints : dict
            The constraints list to be used in the optimization. The keys are the constraints names and the values are the bounds.

        lambda_bound : int
            Lower and upper bound for the scoring system cofficients.
        """
        self.objectives = objectives
        self.fairness_groups = fairness_groups
        self.fairness_labels = fairness_labels
        self.constraints = {} if constraints is None else constraints
        self.lambda_bound = lambda_bound
        self.time_limit = time_limit
        self.verbose = verbose

    def fit(self, X, y):
        N_class = get_class_count(y)
        class_indexes = get_class_indexes(y)

        self.lambdas = self.solve_model(X, y, N_class, class_indexes)
        # accuracy = get_accuracy(X, y, self.lambdas)
        # balanced_accuracy = get_balanced_accuracy(X, y, self.lambdas)

    def solve_model(self, X, y, N_class, class_indexes):
        N = len(X)
        L = len(y[0])
        D = len(X[0])
        gamma = 0.01
        M = self.lambda_bound * D + 1

        import cvxpy as cp

        l = cp.Variable((L, D))
        constraints = [l <= self.lambda_bound, l >= -self.lambda_bound]

        for g in self.fairness_groups:
            constraint = l[:, g] == 0
            constraints.append(constraint)

        z = cp.Variable(N, boolean=True)

        if "s" in self.constraints:
            alpha = cp.Variable((L, D), boolean=True)

        if (
            "omr" in self.constraints
            or "sp" in self.constraints
            or "pe" in self.constraints
            or "eo" in self.constraints
            or "eod" in self.constraints
        ):
            pos = cp.Variable((N, L), boolean=True)

        y_idx = np.argmax(y, axis=1)
        if "a" in self.objectives or "ba" in self.objectives:
            for i in range(N):
                for offset in range(1, L):
                    new_contrant = -M * z[i] <= cp.sum(l[y_idx[i], :] @ X[i, :])
                    -gamma * y_idx[i]
                    -cp.sum(l[(y_idx[i] + offset) % L, :] @ X[i, :])
                    -gamma * ((y_idx[i] + offset) % L)
                    constraints.append(new_contrant)

        if "s" in self.constraints:
            constraints.append(-self.lambda_bound * alpha <= l)
            constraints.append(l <= self.lambda_bound * alpha)

        if (
            "omr" in self.constraints
            or "sp" in self.constraints
            or "pe" in self.constraints
            or "eo" in self.constraints
            or "eod" in self.constraints
        ):
            for i in range(N):
                for index in range(L):
                    for offset in range(1, L):
                        new_constaint = -M * (1 - pos[i, index]) <= cp.sum(l[index, :] @ X[i, :])
                        -gamma * index
                        -cp.sum(l[(index + offset) % L, :] @ X[i, :])
                        -gamma * ((index + offset) % L)
                        constraints.append(new_constaint)
                constraints.append(cp.sum(pos[i, :]) == 1)

        if "s" in self.constraints:
            for index in range(L):
                contraint = cp.sum(alpha[index, :]) <= self.constraints["s"]
                constraints.append(contraint)

        for index in self.fairness_labels:
            for g in self.fairness_groups:
                i_g = [i for i in range(N) if X[i][g] == 1]
                i_g_bar = [i for i in range(N) if X[i][g] == 0]
                N_g = len(i_g)
                N_g_bar = len(i_g_bar)

                i_g_pos = [i for i in range(N) if X[i][g] == 1 and y[i][index] == 1]
                i_g_neg = [i for i in range(N) if X[i][g] == 1 and y[i][index] == 0]
                N_g_pos = len(i_g_pos)
                N_g_neg = len(i_g_neg)

                i_g_bar_pos = [i for i in range(N) if X[i][g] == 0 and y[i][index] == 1]
                i_g_bar_neg = [i for i in range(N) if X[i][g] == 0 and y[i][index] == 0]
                N_g_bar_pos = len(i_g_bar_pos)
                N_g_bar_neg = len(i_g_bar_neg)

                if N_g == 0 or N_g_bar == 0:
                    logger.info("At least one of the protected groups is empty, skipping fairness constraints")
                    break

                if "omr" in self.constraints:
                    constraints.append(
                        -self.constraints["omr"]
                        <= (1 / N_g) * cp.sum(pos[i_g_neg, index])
                        + cp.sum(1 - pos[i_g_pos, index])
                        - (1 / N_g_bar) * cp.sum(pos[i_g_bar_neg, index])
                        + cp.sum(1 - pos[i_g_bar_pos, index])
                    )
                    constraints.append(
                        (1 / N_g) * cp.sum(pos[i_g_neg, index])
                        + cp.sum(1 - pos[i_g_pos, index])
                        - (1 / N_g_bar) * cp.sum(pos[i_g_bar_neg, index])
                        + cp.sum(1 - pos[i_g_bar_pos, index])
                        <= self.constraints["omr"]
                    )

                if "sp" in self.constraints and i_g and i_g_bar:
                    constraints.append(
                        -self.constraints["sp"]
                        <= (1 / N_g) * cp.sum(pos[i_g, index]) - (1 / N_g_bar) * cp.sum(pos[i_g_bar, index])
                    )
                    constraints.append(
                        (1 / N_g) * cp.sum(pos[i_g, index]) - (1 / N_g_bar) * cp.sum(pos[i_g_bar, index])
                        <= self.constraints["sp"]
                    )

                if "pe" in self.constraints and i_g_neg and i_g_bar_neg:
                    constraints.append(
                        -self.constraints["pe"]
                        <= (1 / N_g_neg) * cp.sum(pos[i_g_neg, index])
                        - (1 / N_g_bar_neg) * cp.sum(pos[i_g_bar_neg, index])
                    )
                    constraints.append(
                        (1 / N_g_neg) * cp.sum(pos[i_g_neg, index])
                        - (1 / N_g_bar_neg) * cp.sum(pos[i_g_bar_neg, index])
                        <= self.constraints["pe"]
                    )

                if "eo" in self.constraints and i_g_pos and i_g_bar_pos:
                    constraints.append(
                        -self.constraints["eo"]
                        <= (1 / N_g_pos) * cp.sum(pos[i_g_pos, index])
                        - (1 / N_g_bar_pos) * cp.sum(pos[i_g_bar_pos, index])
                    )
                    constraints.append(
                        (1 / N_g_pos) * cp.sum(pos[i_g_pos, index])
                        - (1 / N_g_bar_pos) * cp.sum(pos[i_g_bar_pos, index])
                        <= self.constraints["eo"]
                    )

                if "eod" in self.constraints and i_g_neg and i_g_bar_neg and i_g_pos and i_g_bar_pos:
                    constraints.append(
                        -self.constraints["eod"]
                        <= (1 / N_g_neg) * cp.sum(pos[i_g_neg, index])
                        - (1 / N_g_bar_neg) * cp.sum(pos[i_g_bar_neg, index])
                    )
                    constraints.append(
                        (1 / N_g_neg) * cp.sum(pos[i_g_neg, index])
                        - (1 / N_g_bar_neg) * cp.sum(pos[i_g_bar_neg, index])
                        <= self.constraints["eod"]
                    )
                    constraints.append(
                        -self.constraints["eod"]
                        <= (1 / N_g_pos) * cp.sum(pos[i_g_pos, index])
                        - (1 / N_g_bar_pos) * cp.sum(pos[i_g_bar_pos, index])
                    )
                    constraints.append(
                        (1 / N_g_pos) * cp.sum(pos[i_g_pos, index])
                        - (1 / N_g_bar_pos) * cp.sum(pos[i_g_bar_pos, index])
                        <= self.constraints["eod"]
                    )

        if "a" in self.objectives and "s" not in self.constraints:
            cost = (1 / N) * cp.sum(z)

        if "ba" in self.objectives and "s" not in self.constraints:
            cost = 0
            for i, indexes in enumerate(class_indexes):
                cost += cp.sum(z[indexes]) / N_class[i]
            cost /= len(N_class)

        if "a" in self.objectives and "s" in self.constraints:
            cost = (1 / N) * cp.sum(z) + (1 / (self.constraints["s"] * L * N)) * cp.sum(alpha)

        if "ba" in self.objectives and "s" in self.constraints:
            cost = 0
            for i, indexes in enumerate(class_indexes):
                cost += cp.sum(z[indexes]) / N_class[i]
            cost /= len(N_class)
            cost += (1 / (self.constraints["s"] * L * N)) * cp.sum(alpha)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.CBC, verbose=self.verbose == 1, maximumSeconds=self.time_limit)
        return l.value

    def predict(self, X):
        """
        Returns the predictions of the set of scoring systems for the given entries

        X : The input features matrix
        """

        y = []

        for _, sample in enumerate(X):
            scores = []
            for l_list in self.lambdas:
                score = sum([feature * l_list[j] for j, feature in enumerate(sample)])
                scores.append(score)
            y_pred = []
            for i in range(len(scores)):
                if i == np.argmax(scores):
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            y.append(y_pred)

        return np.array(y)
