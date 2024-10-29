import numpy as np
from scipy import stats

EPSILON = 1e-8


def get_average_local_importances(local_importances_values: np.ndarray):
    local_importances_values = np.abs(local_importances_values)
    local_importances_values = local_importances_values / local_importances_values.sum(axis=1, keepdims=True)
    average_local_importances_values = local_importances_values.mean(axis=0)
    return average_local_importances_values


def local_normalized_desviation(local_importances_values: np.ndarray):
    ranked_features = np.argsort(np.argsort(-local_importances_values, axis=1), axis=1)
    mode_result = stats.mode(ranked_features, axis=0)
    normalized_desviation = (np.abs(ranked_features - np.array(mode_result.mode)[None, :])) / (
        np.max(ranked_features, axis=0) - np.min(ranked_features, axis=0) + EPSILON
    )[None, :]
    return normalized_desviation


def rank_consistency(local_importances_values: np.ndarray, weighted=False, aggregate=True):
    normalized_desviation = local_normalized_desviation(local_importances_values)
    consistencies = np.mean(normalized_desviation, axis=0)
    if aggregate:
        if weighted:
            average_local_importances_values = get_average_local_importances(local_importances_values)
            return np.sum(average_local_importances_values * consistencies)
        return np.mean(consistencies)
    return consistencies
