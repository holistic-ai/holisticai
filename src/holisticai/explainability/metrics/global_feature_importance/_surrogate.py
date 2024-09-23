from sklearn.metrics import accuracy_score, mean_squared_error


def surrogate_accuracy_score(y_pred, y_surrogate):
    return accuracy_score(y_pred, y_surrogate)


def surrogate_mean_squared_error(y_pred, y_surrogate):
    return mean_squared_error(y_pred, y_surrogate)
