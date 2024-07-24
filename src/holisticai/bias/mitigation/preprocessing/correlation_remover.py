import jax.numpy as jnp
from holisticai.utils.transformers.bias import BMPreprocessing as BMPre
from jax import jit


@jit
def _fit(X, sensitive_features, sensitive_mean):
    sensitive_features_center = sensitive_features - sensitive_mean
    beta = jnp.linalg.lstsq(sensitive_features_center, X, rcond=None)[0]
    return beta


@jit
def _transform(X, sensitive_features, beta, alpha, sensitive_mean):
    sensitive_features_center = sensitive_features - sensitive_mean
    x_filtered = X - jnp.dot(sensitive_features_center, beta)
    return alpha * x_filtered + (1 - alpha) * X


class CorrelationRemover(BMPre):
    """
    CorrelationRemover applies a linear transformation to the non-sensitive feature columns\
    in order to remove their correlation with the sensitive feature columns while retaining\
    as much information as possible (as measured by the least-squares error).

    Parameters
    ----------
    alpha : float, optional
        parameter to control how much to filter, for alpha=1.0 we filter out
        all information while for alpha=0.0 we don't apply any. Default is 1.

    Notes
    -----
    This method will change the original dataset by removing all correlation with sensitive\
    values. Note that the lack of correlation does not imply anything about statistical dependence.\
    Therefore, it is expected this to be most appropriate as a preprocessing step for\
    (generalized) linear models.
    """

    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X: jnp.ndarray, group_a: jnp.ndarray, group_b: jnp.ndarray):
        """
        Learn the projection required to make the dataset uncorrelated with groups (group_a and group_b).

        Parameters
        ----------
        X : matrix-like
            Input data
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        ------
        Self
        """
        params = self._load_data(X=X, group_a=group_a, group_b=group_b)
        x = params["X"]
        group_a = params["group_a"]
        group_b = params["group_b"]

        sensitive_features = jnp.stack([group_a, group_b], axis=1).astype(jnp.int32)
        self.sensitive_mean_ = jnp.mean(sensitive_features, axis=0)
        self.beta_ = _fit(x, sensitive_features, self.sensitive_mean_)
        self.x_shape_ = x.shape
        return self

    def transform(self, X: jnp.ndarray, group_a: jnp.ndarray, group_b: jnp.ndarray):
        """
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
        x = params["X"]
        group_a = params["group_a"]
        group_b = params["group_b"]

        sensitive_features = jnp.stack([group_a, group_b], axis=1).astype(jnp.int32)
        x_filtered = _transform(x, sensitive_features, self.beta_, self.alpha, self.sensitive_mean_)
        return x_filtered

    def fit_transform(self, X: jnp.ndarray, group_a: jnp.ndarray, group_b: jnp.ndarray):
        """
        Fit and transform

        Parameters
        ----------
        X : matrix-like
            Input data
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        ------
            Self
        """
        self.fit(X, group_a, group_b)
        return self.transform(X, group_a, group_b)
