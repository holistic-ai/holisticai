import numpy as np
import numpy.linalg as la


def adversarial_accuracy(
    y: np.ndarray,
    y_pred: np.ndarray,
    y_adv_pred: np.ndarray,
) -> float:
    """
    Calculate the adversarial accuracy of a model.

    Parameters
    ----------
    y : np.ndarray
        The true labels. If `None`, the function calculates the accuracy based on `y_pred` and `y_adv_pred` directly.
    y_pred : np.ndarray
        The predicted labels for the original input.
    y_adv_pred : np.ndarray
        The predicted labels for the adversarial input.

    Returns
    -------
    float
        The adversarial accuracy value.

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.robustness.metrics import adversarial_accuracy
    >>> y = np.array([1, 0, 1, 1])
    >>> y_pred = np.array([1, 0, 0, 1])
    >>> y_adv_pred = np.array([0, 0, 1, 1])
    >>> adversarial_accuracy(y, y_pred, y_adv_pred)
    0.6666666666666666
    """
    if y is None:
        idxs = y_pred == y_adv_pred
        return np.sum(idxs) / len(y_adv_pred)

    # Verify that all inputs are of type np.ndarray
    if not all(isinstance(arg, np.ndarray) for arg in [y, y_pred, y_adv_pred]):
        raise TypeError("All arguments must be of type np.ndarray")

    y_corr = y_pred == y
    return np.sum((y_pred == y_adv_pred) & y_corr) / np.sum(y_corr)


def empirical_robustness(
    x: np.ndarray,
    adv_x: np.ndarray,
    y_pred: np.ndarray,
    y_adv_pred: np.ndarray,
    norm: int = 2,
) -> float:
    """
    Calculate the empirical robustness of an adversarial example.

    Parameters
    ----------
    x : np.ndarray
        The original input.
    adv_x : np.ndarray
        The adversarial input.
    y_pred : np.ndarray
        The predicted labels for the original input.
    y_adv_pred : np.ndarray
        The predicted labels for the adversarial input.
    norm : int (optional)
        The norm to be used for calculating the perturbation. Defaults to 2.

    Returns
    -------
    float
        The empirical robustness value.

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.robustness.metrics import empirical_robustness
    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> adv_x = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    >>> y_pred = np.array([0, 1])
    >>> y_adv_pred = np.array([1, 1])
    >>> empirical_robustness(x, adv_x, y_pred, y_adv_pred)
    0.09999999999999999
    """
    # Verify that the parameters have the correct types
    if not isinstance(norm, int):
        raise TypeError("norm must be of type int")

    if not all(isinstance(arg, np.ndarray) for arg in [x, adv_x, y_pred, y_adv_pred]):
        raise TypeError("x, adv_x, y_pred, and y_adv_pred must be of type np.ndarray")

    idxs = y_adv_pred != y_pred
    if np.sum(idxs) == 0.0:
        return 0.0

    perts_norm = la.norm((adv_x - x).reshape(x.shape[0], -1), ord=norm, axis=1)
    perts_norm = perts_norm[idxs]

    return np.mean(perts_norm / la.norm(x[idxs].reshape(np.sum(idxs), -1), ord=norm, axis=1))
