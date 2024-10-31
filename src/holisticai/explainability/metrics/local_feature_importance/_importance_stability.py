import numpy as np


def importance_stability(feature_importances: np.ndarray, aggregate=True):
    """

    Determine the stability of feature importance.

    Parameters
    ----------

    feature_importances: np.array
    A matrix of shape (M, d), where M is the number of samples and d is the number
    of features. Each entry represents the importance of a feature in a sample.

    Returns
    -------

    stability: float
    The stability metric, bounded between 0 and 1.
    """
    M, d = feature_importances.shape  # M: number of samples, d: number of features

    feature_importances = feature_importances / np.sum(feature_importances, axis=1)[:, np.newaxis]
    # Mean importance for each feature
    mean_importances = np.mean(feature_importances, axis=0)
    mean_importances_norm = mean_importances / mean_importances.sum()

    # Variance of the importance for each feature
    var_importances = np.var(feature_importances, axis=0, ddof=1)  # ddof=1 for sample variance

    # Calculate stability
    stabilities = []
    for j in range(d):
        if mean_importances[j] == 0 or mean_importances[j] == 1:
            stability_j = 0  # If mean is 0 or 1, it is perfectly stable for that feature.
        else:
            # Variance normalized by mean and (1 - mean)
            stability_j = var_importances[j] / (mean_importances[j] * (1 - mean_importances[j]))
        score = float(max(0, min(1, stability_j)))
        stabilities.append(score)

    if aggregate:
        return np.sum([mean_importances_norm[j] * stabilities[j] for j in range(d)])
    return stabilities
