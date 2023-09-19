import numpy as np
from scipy.optimize import fmin_cg

from holisticai.utils.transformers.bias import SensitiveGroups


class PrejudiceRemoverAlgorithm:
    """Two class LogisticRegression with Prejudice Remover"""

    def __init__(self, estimator, objective_fn, logger, maxiter=None):
        """
        Parameters
        ----------
        estimator : estimator object
            Prejudice Remover Model

        objective_fn : object
            Objective class with loss and grad_loss function

        logger : object
            Support for print information

        maxiter: int
            Maximum number of iterations for nonlinear conjugate gradient algorithm. Default is ``200 * len(x0)``.

        """

        self.estimator = estimator
        self.logger = logger
        self.objective_fn = objective_fn
        self.maxiter = maxiter
        self.sens_groups = SensitiveGroups()

    def fit(self, X: np.ndarray, y: np.ndarray, sensitive_features: np.ndarray):
        """
        Description
        -----------
        Optimize the model paramters to reduce the loss function

        Parameters
        ----------
        X : matrix-like
            Input matrix

        y_true : numpy array
            Target vector

        sensitive_features : numpy array
            Matrix where each columns is a sensitive feature e.g. [col_1=group_a, col_2=group_b]
        """

        groups_num = self.sens_groups.fit_transform(
            sensitive_features, convert_numeric=True
        )
        self.estimator.init_params(X, y, groups_num)
        self.logger.set_log_fn(
            loss=lambda coef: self.objective_fn.loss(coef, X, y, groups_num), type=float
        )

        self.coef = fmin_cg(
            self.objective_fn.loss,
            self.estimator.coef,
            fprime=self.objective_fn.grad_loss,
            args=(X, y, groups_num),
            maxiter=self.maxiter,
            disp=False,
            callback=self.logger.callback,
        )
        self.estimator.set_params(self.coef)

        self.f_loss_ = self.objective_fn.loss(self.coef, X, y, groups_num)
        self.logger.info(f"Best Loss : {self.f_loss_:.4f}")
        return self

    def predict(self, X, sensitive_features):
        """
        predict classes

        Parameters
        ----------
        X : array, shape=(n_samples, n_features)
            feature vectors of samples

        sensitive_features : numpy array
            Matrix where each columns is a sensitive feature e.g. [col_1=group_a, col_2=group_b]
        Returns
        -------
        y : array, shape=(n_samples), dtype=int
            array of predicted class
        """
        p_attr = self.sens_groups.transform(sensitive_features, convert_numeric=True)
        return self.estimator.predict(X, p_attr)

    def predict_proba(self, X, sensitive_features):
        """
        Predict probabilities

        Parameters
        ----------
        X : array, shape=(n_samples, n_features)
            feature vectors of samples

        sensitive_features : numpy array
            Matrix where each columns is a sensitive feature e.g. [col_1=group_a, col_2=group_b]

        Returns
        -------
        y_proba : array, shape=(n_samples, n_classes), dtype=float
            array of predicted class
        """
        p_attr = self.sens_groups.transform(sensitive_features, convert_numeric=True)
        return self.estimator.predict_proba(X, p_attr)
