import numpy as np
import numpy.linalg as la


def adversarial_accuracy(y, y_pred, y_adv_pred):
    """
    Calculate the adversarial accuracy of a model with array-like inputs.

    Parameters
    ----------
    y : array-like
        The true labels. If `None`, the function calculates the accuracy based on `y_pred` and `y_adv_pred` directly.
    y_pred : array-like
        The predicted labels for the original input.
    y_adv_pred : array-like
        The predicted labels for the adversarial input.

    Returns
    -------
    float
        The adversarial accuracy value.

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.robustness.metrics import adversarial_accuracy
    >>> y = [1, 0, 1, 1]  # Example with list input
    >>> y_pred = [1, 0, 0, 1]
    >>> y_adv_pred = [0, 0, 1, 1]
    >>> adversarial_accuracy(y, y_pred, y_adv_pred)
    0.6666666666666666
    """
    try:
        # Convert inputs to np.ndarray if they are not already
        y = np.array(y) if y is not None else None
        y_pred = np.array(y_pred)
        y_adv_pred = np.array(y_adv_pred)
    except Exception as e:
        raise ValueError("One or more inputs could not be converted to a numpy array.") from e

    if y is None:
        idxs = y_pred == y_adv_pred
        return np.sum(idxs) / len(y_adv_pred)

    y_corr = y_pred == y
    return np.sum((y_pred == y_adv_pred) & y_corr) / np.sum(y_corr)


def empirical_robustness(
    x,
    adv_x,
    y_pred,
    y_adv_pred,
    norm=2,
) -> float:
    """
    Calculate the empirical robustness of an adversarial example.

    Parameters
    ----------
    x : array-like
        The original input.
    adv_x : array-like
        The adversarial input.
    y_pred : array-like
        The predicted labels for the original input.
    y_adv_pred : array-like
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
    try:
        x = np.array(x)
        adv_x = np.array(adv_x)
        y_pred = np.array(y_pred)
        y_adv_pred = np.array(y_adv_pred)
    except Exception as e:
        raise ValueError("One or more inputs could not be converted to a numpy array.") from e

    # Verify that `norm` has the correct type
    if not isinstance(norm, int):
        raise TypeError("norm must be of type int")

    idxs = y_adv_pred != y_pred
    if np.sum(idxs) == 0.0:
        return 0.0

    perts_norm = la.norm((adv_x - x).reshape(x.shape[0], -1), ord=norm, axis=1)
    perts_norm = perts_norm[idxs]

    return np.mean(perts_norm / la.norm(x[idxs].reshape(np.sum(idxs), -1), ord=norm, axis=1))
