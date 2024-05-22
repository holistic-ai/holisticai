from lib2to3.pgen2.token import OP
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from holisticai.utils.transformers.bias import BMInprocessing as BMImp

from .algorithm import PrejudiceRemoverAlgorithm
from .algorithm_utils import ObjectiveFunction, PRLogger
from .losses import PRBinaryCrossEntropy
from .model import PRLogiticRegression, PRParamInitializer


class PrejudiceRemover(BaseEstimator, ClassifierMixin, BMImp):
    """
    Prejudice remover is an in-processing technique that adds a
    discrimination-aware regularization term to the learning objective.

    References
    ----------
        Kamishima, Toshihiro, et al. "Fairness-aware classifier with prejudice remover regularizer."
        Joint European conference on machine learning and knowledge discovery in databases.
        Springer, Berlin, Heidelberg, 2012.
    """

    def __init__(
        self,
        eta: Optional[float] = 1.0,
        C: Optional[float] = 1.0,
        fit_intercept: Optional[bool] = True,
        penalty: Optional[str] = "l2",
        init_type: Optional[str] = "Zero",
        maxiter: Optional[int] = 1000,
        verbose: Optional[int] = 0,
        print_interval: Optional[int] = 20,
    ):
        """
        Parameters
        ----------
            eta : float
                fairness penalty parameter

            C : float
                Inverse of regularization strength (same as sklearn).

            fit_intercept : bool
                Specifies if a constant must be added to the decision function (same as sklearn).

            penalty : str
                Specify the norm of the penalty (same as sklearn).

            init_type : str
                Specifies how the model parameters will be initialized:
                - Zero : Set all model parameters with zero value.
                - Random : Initialize model parameters with random values.
                - StandarLR : Initialize model parameters with fitted sklearn LR.
                - StandarLRbyGroup : Initialize model parameters with fitted sklearn LR for group_a and group_b.

            maxiter : str
                Maximum number of iterations.

            verbose : int
                Log progress if value > 0.

            print_interval : int
                Each `print_interval` steps print information.

        """

        # Default estimator parameters
        self.eta = eta
        self.C = C
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.init_type = init_type

        self.maxiter = maxiter
        self.verbose = verbose
        self.print_interval = print_interval

    def transform_estimator(self):
        # Regularized Binary Cross Entropy
        loss_ = PRBinaryCrossEntropy(C=self.C, eta=self.eta)

        # Model Parameter Initializer
        initializer_ = PRParamInitializer(
            init_type=self.init_type,
            C=self.C,
            penalty=self.penalty,
            fit_intercept=self.fit_intercept,
        )

        # Model
        self.estimator = PRLogiticRegression(
            initializer=initializer_, loss=loss_, fit_intercept=self.fit_intercept
        )
        return self

    def _build_algorithm(self):
        # Support class for logger
        logger_ = PRLogger(
            total_iterations=self.maxiter,
            verbose=self.verbose,
            print_interval=self.print_interval,
        )

        # Support class for objective Function
        objective_fn_ = ObjectiveFunction(
            estimator=self.estimator, loss_fn=self.estimator.loss
        )

        # Algorithm class
        algorithm = PrejudiceRemoverAlgorithm(
            estimator=self.estimator,
            logger=logger_,
            objective_fn=objective_fn_,
            maxiter=self.maxiter,
        )

        return algorithm

    def fit(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Fit the model

        Description
        -----------
        Learns the regularized logistic regression model.

        Parameters
        ----------
            X : ndarray
                input data

            y_true : ndarray
                Target vector

            group_a : ndarray
                group mask

            group_b : ndarray
                group mask
        Returns:
            Self
        """
        params = self._load_data(X=X, y_true=y_true, group_a=group_a, group_b=group_b)
        X = params["X"]
        y_true = params["y_true"]
        group_b = params["group_b"]
        group_a = params["group_a"]
        sensitive_features = np.stack([group_a, group_b], axis=1)

        self.classes_ = params["classes_"]

        self.algorithm = self._build_algorithm()
        self.algorithm.fit(X=X, y=y_true, sensitive_features=sensitive_features)
        return self

    def predict(self, X: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Prediction

        Description
        ----------
        Predict output for the given samples.

        Parameters
        ----------
        X : pandas.DataFrame or numpy array
            Test samples.

        group_a : ndarray
                group mask

        group_b : ndarray
                group mask

        Returns
        -------

        numpy.ndarray: Predicted output per sample.
        """
        params = self._load_data(X=X, group_a=group_a, group_b=group_b)
        X = params["X"]
        group_b = params["group_b"]
        group_a = params["group_a"]
        sensitive_features = np.stack([group_a, group_b], axis=1)
        return self.algorithm.predict(X, sensitive_features)

    def predict_proba(self, X: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Prediction

        Description
        ----------
        Predict output for the given samples.

        Parameters
        ----------
        X : numpy array
            Test samples.

        group_a : ndarray
                group mask

        group_b : ndarray
                group mask

        Returns
        -------

        numpy.ndarray: Predicted output per sample.
        """
        params = self._load_data(X=X, group_a=group_a, group_b=group_b)
        X = params["X"]
        group_b = params["group_b"]
        group_a = params["group_a"]
        sensitive_features = np.stack([group_a, group_b], axis=1)
        return self.algorithm.predict_proba(X, sensitive_features)
