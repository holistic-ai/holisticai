import numpy as np
import numpy.linalg as la


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
