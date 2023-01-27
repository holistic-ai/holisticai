import numpy as np
from sklearn.base import TransformerMixin

from holisticai.utils.transformers.bias import BMPreprocessing as BMPre


class CorrelationRemover(TransformerMixin, BMPre):
    """
    CorrelationRemover applies a linear transformation to the non-sensitive feature columns
    in order to remove their correlation with the sensitive feature columns while retaining
    as much information as possible (as measured by the least-squares error).

    Notes
    -----
    This method will change the original dataset by removing all correlation with sensitive
    values. Note that the lack of correlation does not imply anything about statistical dependence.
    Therefore, it is expected this to be most appropriate as a preprocessing step for
    (generalized) linear models.

    Reference
    ---------

    """

    def __init__(self, alpha=1):
        """
        Parameters
        ----------
            alpha : float
                parameter to control how much to filter, for alpha=1.0 we filter out
                all information while for alpha=0.0 we don't apply any.
        """
        self.alpha = alpha

    def fit(self, X: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):
        """
        Fit.

        Description
        -----------
        Learn the projection required to make the dataset uncorrelated with groups (group_a and group_b).

        Parameters
        ----------
        X : matrix-like
            Input data
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Return
        ------
            Self
        """
        params = self._load_data(X=X, group_a=group_a, group_b=group_b)
        X = params["X"]
        group_a = params["group_a"]
        group_b = params["group_b"]

        sensitive_features = np.stack([group_a, group_b], axis=1).astype(np.int32)
        self.sensitive_mean_ = sensitive_features.mean()
        sensitive_features_center = sensitive_features - self.sensitive_mean_
        self.beta_, _, _, _ = np.linalg.lstsq(sensitive_features_center, X, rcond=None)
        self.X_shape_ = X.shape
        return self

    def transform(self, X: np.ndarray, group_a: np.ndarray, group_b: np.ndarray):

        """
        Description
        ----------
        Transform X by applying the correlation remover.

        Parameters
        ----------
        X : matrix-like
            Input matrix
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        Returns
        -------
            np.ndarray
        """

        params = self._load_data(X=X, group_a=group_a, group_b=group_b)
        X = params["X"]
        group_a = params["group_a"]
        group_b = params["group_b"]

        sensitive_features = np.stack([group_a, group_b], axis=1).astype(np.int32)
        self.sensitive_mean_ = sensitive_features.mean()
        sensitive_features_center = sensitive_features - self.sensitive_mean_
        X_filtered = X - sensitive_features_center.dot(self.beta_)
        X = np.atleast_2d(X)
        X_filtered = np.atleast_2d(X_filtered)
        return self.alpha * X_filtered + (1 - self.alpha) * X

    def fit_transform(
        self,
        X: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Fit and transform

        Description
        ----------
        Fit and transform

        Parameters
        ----------
        X : matrix-like
            Input data
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Return
        ------
            Self
        """
        return self.fit(X, group_a, group_b).transform(X, group_a, group_b)
