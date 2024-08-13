from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from holisticai.bias.mitigation.inprocessing.fair_scoring_classifier.algorithm import FairScoreClassifierAlgorithm
from holisticai.utils.transformers.bias import BMInprocessing as BMImp
from holisticai.utils.transformers.bias import SensitiveGroups
from sklearn.base import BaseEstimator


class FairScoreClassifier(BaseEstimator, BMImp):
    """Fair Score Classifier [1]_ generates a classification model that integrates fairness constraints for multiclass classification. This algorithm\
    returns a matrix of lambda coefficients that scores a given input vector. The higher the score, the higher the probability\
    of the input vector to be classified as the majority class.

    Parameters
    ----------
        objectives : dict
            The weighted objectives list to be optimized.

        constraints : dict
            The constraints list to be used in the optimization. The keys are the constraints names and the values are the bounds.

        lambda_bound : int
            Lower and upper bound for the scoring system cofficients.

        time_limit : int
            The time limit for the optimization algorithm.

    References:
        .. [1] Julien Rouzot, Julien Ferry, Marie-José Huguet. Learning Optimal Fair Scoring Systems for Multi-\
        Class Classification. ICTAI 2022 - The 34th IEEE International Conference on Tools with Artificial\
        Intelligence, Oct 2022, Virtual, United States.
    """

    def __init__(
        self,
        objectives: Literal["a", "ab"],
        constraints: Optional[dict] = None,
        lambda_bound: int = 9,
        time_limit: int = 100,
        verbose: int = 0,
    ):
        self.objectives = objectives
        self.constraints = {} if constraints is None else constraints
        self.lambda_bound = lambda_bound
        self.time_limit = time_limit
        self.verbose = verbose

    def fit(self, X, y, group_a, group_b):
        """
        Fit model using Grid Search Algorithm.

        Parameters
        ----------

        X : matrix-like
            input matrix

        y : numpy array
            target vector

        protected_groups : list
            The sensitive groups.

        protected_labels : list
            The senstive labels.

        sensitive_features : numpy array
            Matrix where each columns is a sensitive feature e.g. [col_1=group_a, col_2=group_b]

        Returns
        -------
            self
        """
        self._sensgroups = SensitiveGroups()
        groups = np.stack([np.squeeze(group_a), np.squeeze(group_b)], axis=1).reshape([-1, 2])
        p_attr = self._sensgroups.fit_transform(groups, convert_numeric=True)
        Xtrain = np.hstack([np.ones((X.shape[0], 1)), X, p_attr.values.reshape(-1, 1)])
        fairness_groups = [Xtrain.shape[1] - 1]
        y_oh = pd.get_dummies(np.squeeze(y))
        fairness_labels = np.arange(y_oh.shape[1]).tolist()

        self.model_ = FairScoreClassifierAlgorithm(
            self.objectives,
            fairness_groups,
            fairness_labels,
            self.constraints,
            self.lambda_bound,
            self.time_limit,
            self.verbose,
        )
        self.model_.fit(Xtrain, np.array(y_oh))
        return self

    def predict(self, X, group_a, group_b):
        """
        Predict the target vector.

        Parameters
        ----------
        X : matrix-like
            input matrix

        group_a : numpy array
            binary mask vector

        group_b : numpy array
            binary mask vector

        Returns
        -------
            numpy array
        """
        p_attr = self._sensgroups.transform(
            np.stack([np.squeeze(group_a), np.squeeze(group_b)], axis=1), convert_numeric=True
        )
        X_ = np.hstack([np.ones((X.shape[0], 1)), X, p_attr.values.reshape(-1, 1)])
        preds = self.model_.predict(X_)
        y_pred = pd.DataFrame(preds, columns=np.arange(preds.shape[1]))
        return y_pred.idxmax(axis=1).values

    def transform_estimator(self, estimator):
        """
        Transform the estimator.

        Parameters
        ----------
        estimator : object
            The estimator object.

        Returns
        -------
            self
        """
        self.estimator = estimator
        return self
