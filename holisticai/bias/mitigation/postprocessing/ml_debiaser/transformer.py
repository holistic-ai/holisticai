import numpy as np

from holisticai.utils.transformers.bias import BMPostprocessing as BMPost
from holisticai.utils.transformers.bias import SensitiveGroups

from .randomized_threshold.algorithm import RandomizedThresholdAlgorithm
from .reduce2binary.algorithm import Reduce2BinaryAlgorithm


class MLDebiaser(BMPost):
    """
    MLDebiaser postprocessing debias predictions w.r.t. the sensitive class in
    each demographic group. This procedure takes as input a vector y and solves
    the optimization problem subject to the statistical parity constraint. This
    bias mitigation can be used for classification (binary and multiclass).

    Reference
    ---------
        Alabdulmohsin, Ibrahim M., and Mario Lucic. "A near-optimal algorithm for debiasing
        trained machine learning models." Advances in Neural Information Processing Systems
        34 (2021): 8072-8084.
    """

    def __init__(
        self,
        gamma=1.0,
        eps=0,
        eta=0.5,
        sgd_steps=10_000,
        full_gradient_epochs=1_000,
        batch_size=256,
        max_iter=5,
        verbose=True,
    ):
        self.gamma = gamma
        self.eps = eps
        self.eta = eta
        self.sgd_steps = sgd_steps
        self.full_gradient_epochs = full_gradient_epochs
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.verbose = verbose
        self.sens_groups = SensitiveGroups()

    def fit(self, y_proba: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Compute parameters for calibrated equalized odds.

        Description
        ----------
        Compute parameters for calibrated equalized odds algorithm.

        Parameters
        ----------
        y_pred : array-like
            Predicted vector (num_examples,).
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        Returns
        -------
        Self
        """
        params = self._load_data(y_proba=y_proba, group_a=group_a, group_b=group_b)
        y_proba = params["y_proba"]
        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1
        num_classes = y_proba.shape[1]
        sensitive_features = np.stack([group_a, group_b], axis=1)
        self.sens_groups.fit(sensitive_features)

        if num_classes > 2:
            self.algorithm = Reduce2BinaryAlgorithm(
                gamma=self.gamma,
                eps=self.eps,
                eta=self.eta,
                num_classes=num_classes,
                sgd_steps=self.sgd_steps,
                full_gradient_epochs=self.full_gradient_epochs,
                batch_size=self.batch_size,
                max_iter=self.max_iter,
                verbose=self.verbose,
            )
        else:
            self.algorithm = RandomizedThresholdAlgorithm(
                gamma=self.gamma,
                eps=self.eps,
                sgd_steps=self.sgd_steps,
                full_gradient_epochs=self.full_gradient_epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
            )
        return self

    def transform(
        self,
        y_proba: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Apply transform function to predictions and likelihoods

        Description
        ----------
        Use a fitted probability to change the output label and invert the likelihood

        Parameters
        ----------
        y_proba : array-like
            Predicted probability matrix (nb_examlpes, nb_classes)
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        threshold : float
            float value to discriminate between 0 and 1

        Returns
        -------
        dictionnary with new predictions
        """
        params = self._load_data(y_proba=y_proba, group_a=group_a, group_b=group_b)

        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1
        y_proba = params["y_proba"]
        sensitive_features = np.stack([group_a, group_b], axis=1)
        p_attr = self.sens_groups.transform(sensitive_features, convert_numeric=True)

        if type(self.algorithm) is Reduce2BinaryAlgorithm:
            # Multiclass classification
            new_y_prob = self.algorithm.predict(y_proba, p_attr)
            new_y_pred = new_y_prob.argmax(axis=-1)
            return {"y_pred": new_y_pred, "y_proba": new_y_prob}
        else:
            # Binary classification
            pred = y_proba[:, 1]  # .argmax(axis=-1)
            pred = (
                2 * pred - 1
            )  # follow author implementation (use prediction and not logit)
            self.algorithm.fit(pred, p_attr)

            new_y_score = self.algorithm.predict(pred, p_attr)
            new_y_pred = np.where(new_y_score > 0.5, 1, 0)
            return {"y_pred": new_y_pred, "y_score": new_y_score}

    def fit_transform(
        self, y_proba: np.ndarray, group_a: np.ndarray, group_b: np.ndarray
    ):
        """
        Fit and transform

        Description
        ----------
        Fit and transform

        Parameters
        ----------
        y_proba : array-like
            Predicted vector (num_examples,).
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        Returns
        -------
        dictionnary with new predictions
        """
        return self.fit(
            y_proba,
            group_a,
            group_b,
        ).transform(y_proba, group_a, group_b)
