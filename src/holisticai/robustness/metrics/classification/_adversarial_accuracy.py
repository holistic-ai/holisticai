import numpy as np


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
