from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from holisticai.bias.mitigation.inprocessing.fair_scoring_classifier.utils import (
    get_class_count,
    get_class_indexes,
)
from scipy.optimize import Bounds, LinearConstraint, milp

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
        num_threads: int = 1,
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
        self.num_threads = num_threads
        self.verbose = verbose

    def fit(self, X, y):
        N_class = get_class_count(y)
        class_indexes = get_class_indexes(y)

        self.lambdas = self.solve_model(X, y, N_class, class_indexes)
        # accuracy = get_accuracy(X, y, self.lambdas)
        # balanced_accuracy = get_balanced_accuracy(X, y, self.lambdas)

    def solve_model(self, X, y, N_class, class_indexes):
        """
        Solve the optimization problem to find the lambda coefficients that minimize the loss function.
        Adapted from the original code in `paper <https://gitlab.laas.fr/roc/julien-rouzot/fairscoringsystemsv0>`_ .

        Parameters
        ----------
        X : np.array
            The input features matrix.

        y : np.array
            The target labels matrix.

        N_class : list
            The number of samples in each class.

        class_indexes : list
            The indexes of each class.
        """
        N, D = X.shape  # Number of samples, features
        _, L = y.shape  # Number of labels
        gamma = 0.01  # Margin parameter
        M = self.lambda_bound * D + 1  # Upper bound on the max score

        if "ba" in self.objectives:
            N_class = np.sum(y, axis=0)  # Get the number of samples in each class
            class_indexes = [np.where(y[:, i] == 1)[0] for i in range(L)]  # Get the indexes of each class

        # Define variables
        num_lambda_vars = L * D
        l = np.arange(num_lambda_vars)
        z = np.arange(num_lambda_vars, num_lambda_vars + N)

        if "s" in self.constraints:
            alpha = np.arange(z[-1] + 1, z[-1] + 1 + num_lambda_vars)

        pos = np.array([]) # Initialize as an empty array
        if any(c in self.constraints for c in ["omr", "sp", "pe", "eo", "eod"]):
            pos = np.arange(alpha[-1] + 1 if "s" in self.constraints else z[-1] + 1, alpha[-1] + 1 + N * L if "s" in self.constraints else z[-1] + 1 + N * L)

        # Define constraints
        A = []
        b = []
        constraint_types = []

        # Constraints for misclassifications
        if "a" in self.objectives or "ba" in self.objectives:
            for i in range(N):
                for j in range(L):
                    if y[i, j] == 1:
                        for k in range(1, L):
                            # Constraint for correct classification
                            coeffs = np.zeros(len(l) + len(z))
                            coeffs[l[j * D:(j + 1) * D]] = X[i]

                            # Correct the indexing here:
                            next_label_index = ((j + k) % L) * D
                            coeffs[l[next_label_index:next_label_index + D]] = -X[i]

                            coeffs[z[i]] = -M
                            A.append(coeffs)
                            b.append(-gamma * j - gamma * ((j + k) % L))
                            constraint_types.append("leq")

        # Constraints for non-zero lambda coefficients
        if "s" in self.constraints:
            for i in range(num_lambda_vars):
                # Constraint: -lambda_bound * alpha <= l
                coeffs = np.zeros(len(l) + len(z) + len(alpha))
                coeffs[l[i]] = 1
                coeffs[alpha[i]] = self.lambda_bound
                A.append(coeffs)
                b.append(0)
                constraint_types.append("leq")

                # Constraint: l <= lambda_bound * alpha
                coeffs = np.zeros(len(l) + len(z) + len(alpha))
                coeffs[l[i]] = -1
                coeffs[alpha[i]] = self.lambda_bound
                A.append(coeffs)
                b.append(0)
                constraint_types.append("leq")

        # Constraints for positive classification
        if any(c in self.constraints for c in ["omr", "sp", "pe", "eo", "eod"]):
            for i in range(N):
                for j in range(L):
                    for k in range(1, L):
                        # Constraint for positive classification
                        coeffs = np.zeros(len(l) + len(z) + (len(alpha) if "s" in self.constraints else 0) + len(pos))
                        coeffs[l[j * D:(j + 1) * D]] = X[i]
                        coeffs[l[((j + k) % L) * D:((j + k + 1) % L) * D]] = -X[i]
                        coeffs[pos[i * L + j]] = M
                        A.append(coeffs)
                        b.append(M - gamma * j + gamma * ((j + k) % L))
                        constraint_types.append("leq")
                # Constraint for one positive classification per sample
                coeffs = np.zeros(len(l) + len(z) + (len(alpha) if "s" in self.constraints else 0) + len(pos))
                coeffs[pos[i * L:(i + 1) * L]] = 1
                A.append(coeffs)
                b.append(1)
                constraint_types.append("eq")

        # Constraints for fairness
        for label_index in self.fairness_labels:
            for g in self.fairness_groups:
                i_g = np.where(X[:, g] == 1)[0]
                i_g_bar = np.where(X[:, g] == 0)[0]
                N_g = len(i_g)
                N_g_bar = len(i_g_bar)

                if N_g == 0 or N_g_bar == 0:
                    print("[WARNING] At least one of the protected groups is empty, skipping fairness constraints")  # noqa: T201
                    continue

                if "omr" in self.constraints:
                    # Constraint for OMR
                    coeffs = np.zeros(len(l) + len(z) + (len(alpha) if "s" in self.constraints else 0) + len(pos))
                    coeffs[pos[i_g * L + label_index]] = 1 / N_g
                    coeffs[pos[i_g_bar * L + label_index]] = -1 / N_g_bar
                    A.append(coeffs)
                    b.append(self.constraints['omr'])
                    constraint_types.append("leq")

                    A.append(-coeffs)
                    b.append(self.constraints['omr'])
                    constraint_types.append("leq")

                if "sp" in self.constraints and N_g and N_g_bar:
                    # Constraint for SP
                    coeffs = np.zeros(len(l) + len(z) + (len(alpha) if "s" in self.constraints else 0) + len(pos))
                    coeffs[pos[i_g * L + label_index]] = 1 / N_g
                    coeffs[pos[i_g_bar * L + label_index]] = -1 / N_g_bar
                    A.append(coeffs)
                    b.append(self.constraints['sp'])
                    constraint_types.append("leq")

                    A.append(-coeffs)
                    b.append(self.constraints['sp'])
                    constraint_types.append("leq")

                # Add constraints for other fairness metrics similarly...

        # Define objective
        c = np.zeros(len(l) + len(z) + (len(alpha) if "s" in self.constraints else 0) + len(pos))
        if "a" in self.objectives:
            c[z] = 1 / N
        if "ba" in self.objectives:
            for i, indexes in enumerate(class_indexes):
                c[z[indexes]] += 1 / (N_class[i] * len(N_class))
        if "s" in self.constraints:
            c[alpha] = 1 / (self.constraints["s"] * L * N)

        # Define bounds
        bounds = [(None if i in l else (0, 1)) for i in range(len(c))]
        if "s" in self.constraints:
            bounds += [(0, 1) for _ in range(len(alpha))]
        if any(c in self.constraints for c in ["omr", "sp", "pe", "eo", "eod"]):
            bounds += [(0, 1) for _ in range(len(pos))]

        # Define integrality constraints
        integrality = np.zeros_like(c)
        integrality[l] = 1
        if "s" in self.constraints:
            integrality[alpha] = 1
        if any(c in self.constraints for c in ["omr", "sp", "pe", "eo", "eod"]):
            integrality[pos] = 1

        # Solve the MILP
        res = milp(
            c=c,
            constraints=[LinearConstraint(A=A, lb=-np.inf, ub=b)],
            bounds=Bounds(lb=-self.lambda_bound, ub=self.lambda_bound),
            integrality=integrality,
        )

        l_lists = []
        if res.success:
            l_values = res.x[:num_lambda_vars]
            for i in range(L):
                l_lists.append(np.rint(l_values[i * D:(i + 1) * D]).astype(int).tolist())  # noqa: PERF401
        else:
            print("[ERROR] Found no solution for this configuration, you may want to soften your constraints")  # noqa: T201

        return l_lists

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
