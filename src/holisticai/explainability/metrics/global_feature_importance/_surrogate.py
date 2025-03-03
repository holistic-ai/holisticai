import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


def surrogate_accuracy_score(y_pred, y_surrogate):
    return accuracy_score(y_pred, y_surrogate)


def surrogate_mean_squared_error(y_pred, y_surrogate):
    return mean_squared_error(y_pred, y_surrogate)


def smape(y_true, y_pred):
    """
    Calculate the Symmetric Mean Absolute Percentage Error (sMAPE)
    Parameters:
    y_true (array-like): Real values
    y_pred (array-like): Predicted values
    Returns:
    float: sMAPE value
    """
    # Convert inputs to numpy arrays for easier calculation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Compute the sMAPE
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    numerator = np.abs(y_true - y_pred)
    # Avoid division by zero by adding a small constant (epsilon) to the denominator
    epsilon = 1e-10
    return np.mean(numerator / (denominator + epsilon))


def surrogate_fidelity(y_pred, y_surrogate):
    return smape(y_pred, y_surrogate)
