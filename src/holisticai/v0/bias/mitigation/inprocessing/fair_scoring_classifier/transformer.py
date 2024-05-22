import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from holisticai.utils.transformers.bias import BMInprocessing as BMImp
from holisticai.utils.transformers.bias import SensitiveGroups

from .algorithm import FairScoreClassifierAlgorithm


class FairScoreClassifier(BaseEstimator, BMImp):
    """
    Generates a classification model that integrates fairness constraints for multiclass classification. This algorithm
    returns a matrix of lambda coefficients that scores a given input vector. The higher the score, the higher the probability
    of the input vector to be classified as the majority class.

    References:
        Julien Rouzot, Julien Ferry, Marie-José Huguet. Learning Optimal Fair Scoring Systems for Multi-
        Class Classification. ICTAI 2022 - The 34th IEEE International Conference on Tools with Artificial
        Intelligence, Oct 2022, Virtual, United States. ￿
    """

    def __init__(
        self,
        objectives: dict,
        constraints: dict = {},
        lambda_bound: int = 9,
        time_limit: int = 100,
    ):
        """
        Init FairScoreClassifier object

        Parameters
        ----------
        objectives : dict
            The weighted objectives list to be optimized.

        constraints : dict
            The constraints list to be used in the optimization. The keys are the constraints names and the values are the bounds.

        lambda_bound : int
            Lower and upper bound for the scoring system cofficients.
        """
        self.objectives = objectives
        self.constraints = constraints
        self.lambda_bound = lambda_bound
        self.time_limit = time_limit

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
        the same object
        """
        sensgroup = SensitiveGroups()
        p_attr = sensgroup.fit_transform(
            np.stack([group_a, group_b], axis=1), convert_numeric=True
        )
        Xtrain = np.hstack([np.ones((X.shape[0], 1)), X, p_attr.values.reshape(-1, 1)])
        fairness_groups = [Xtrain.shape[1] - 1]
        fairness_labels = np.arange(y.shape[1]).tolist()

        self.model_ = FairScoreClassifierAlgorithm(
            self.objectives,
            fairness_groups,
            fairness_labels,
            self.constraints,
            self.lambda_bound,
            self.time_limit,
        )
        self.model_.fit(Xtrain, y)
        return self

    def predict(self, X, group_a, group_b):
        sensgroup = SensitiveGroups()
        p_attr = sensgroup.fit_transform(
            np.stack([group_a, group_b], axis=1), convert_numeric=True
        )
        X_ = np.hstack([np.ones((X.shape[0], 1)), X, p_attr.values.reshape(-1, 1)])
        preds = self.model_.predict(X_)
        y_pred = pd.DataFrame(preds, columns=np.arange(preds.shape[1]))
        return y_pred.idxmax(axis=1).values

    def transform_estimator(self, estimator):
        self.estimator = estimator
        return self
