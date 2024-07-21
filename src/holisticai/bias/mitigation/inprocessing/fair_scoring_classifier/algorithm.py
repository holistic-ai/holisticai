from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from holisticai.bias.mitigation.inprocessing.fair_scoring_classifier.utils import (
    get_class_count,
    get_class_indexes,
    get_initial_solution,
)
from scipy.optimize import minimize

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
        N = len(X)  # Number of samples in the dataset
        L = len(y[0])  # Number of labels in the dataset
        D = len(X[0])  # Number of features for a sample in the dataset
        gamma = 0.01  # Margin parameter
        M = self.lambda_bound * D + 1  # Get an upper bound on the max score

        if "ba" in self.objectives:
            N_class = get_class_count(y)  # Get the number of samples in each class
            class_indexes = get_class_indexes(y)  # Get the indexes of each class

        maj, _ = get_initial_solution(y)  # Get the majority class index

        l = np.zeros((L, D))  # Initialize lambda matrix
        z = np.zeros(N)  # Loss variable

        constraints = []

        if "a" in self.objectives or "ba" in self.objectives:
            for i in range(N):  # Constraint z to model misclassifications
                for index in range(L):
                    if y[i][index] == 1:
                        for offset in range(1, L):
                            constraint = {
                                'type': 'ineq',
                                'fun': lambda l_flat: M * z[i] + np.dot(l_flat[index * D:(index + 1) * D], X[i]) - gamma * index - np.dot(l_flat[((index + offset) % L) * D:((index + offset) % L + 1) * D], X[i]) - gamma * ((index + offset) % L)
                            }
                            constraints.append(constraint)

        if "s" in self.constraints:
            alpha = np.zeros((L, D))  # Alpha matrix
            for index in range(L):  # Constraint alpha to model non-zero lambda coefficients
                for j in range(D):
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda l_flat: self.lambda_bound * alpha[index][j] - l_flat[index * D + j]
                    })
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda l_flat: l_flat[index * D + j] + self.lambda_bound * alpha[index][j]
                    })

        def objective_function(l_flat):
            l = l_flat.reshape(L, D)  # noqa: F841
            if "a" in self.objectives and "s" not in self.constraints:
                return (1 / N) * np.sum(z)  # Minimize loss
            if "ba" in self.objectives and "s" not in self.constraints:
                return np.sum([np.sum(z[indexes]) / N_class[i] for i, indexes in enumerate(class_indexes)]) / len(N_class)  # Minimize balanced loss
            if "a" in self.objectives and "s" in self.constraints:
                return (1 / N) * np.sum(z) + (1 / (self.constraints["s"] * L * N)) * np.sum(alpha)  # Minimize loss and alphas
            if "ba" in self.objectives and "s" in self.constraints:
                return np.sum([np.sum(z[indexes]) / N_class[i] for i, indexes in enumerate(class_indexes)]) / len(N_class) + (1 / (self.constraints["s"] * L * N)) * np.sum(alpha)  # Minimize balanced loss and alphas
            return None

        l_flat = l.flatten()
        sol = minimize(objective_function, l_flat, constraints=constraints)

        if sol.success:
            l_lists = sol.x.reshape(L, D).tolist()
        else:
            print("[ERROR] Found no solution for this configuration, you may want to soften your constraints")  # noqa: T201
            l_lists = []

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
