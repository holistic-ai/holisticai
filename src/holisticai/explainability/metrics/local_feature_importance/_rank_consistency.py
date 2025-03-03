import numpy as np
from scipy import stats

EPSILON = 1e-8


def get_average_local_importances(local_importances_values: np.ndarray):
    local_importances_values = np.abs(local_importances_values)
    local_importances_values = local_importances_values / local_importances_values.sum(axis=1, keepdims=True)
    return local_importances_values.mean(axis=0)


def local_normalized_desviation(local_importances_values: np.ndarray):
    ranked_features = np.argsort(np.argsort(-local_importances_values, axis=1), axis=1)
    mode_result = stats.mode(ranked_features, axis=0)
    return (np.abs(ranked_features - np.array(mode_result.mode)[None, :])) / (
        np.max(ranked_features, axis=0) - np.min(ranked_features, axis=0) + EPSILON
    )[None, :]


def rank_consistency(local_importances_values: np.ndarray, weighted=False, aggregate=True):
    """

    Calculate the rank consistency of local feature importances through all the instances.

    Parameters
    ----------

    local_importances_values : np.ndarray
        A 2D array where each row represents the local feature importances for a specific instance.
    weighted : bool, optional
        If True, the rank consistency will be weighted by the average local importances. Default is False.
    aggregate : bool, optional
        If True, the function will return an aggregated consistency score. If False, it will return the consistency
        scores for each feature. Default is True.

    Returns
    -------

    float or np.ndarray
        If `aggregate` is True, returns a single float representing the aggregated rank consistency.
        If `aggregate` is False, returns an array of consistency scores for each feature.
    """
    normalized_desviation = local_normalized_desviation(local_importances_values)
    consistencies = np.mean(normalized_desviation, axis=0)
    if aggregate:
        if weighted:
            average_local_importances_values = get_average_local_importances(local_importances_values)
            return np.sum(average_local_importances_values * consistencies)
        return np.mean(consistencies)
    return consistencies
