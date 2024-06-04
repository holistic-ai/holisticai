import numpy as np


def convert_float_to_categorical(target, nb_classes, numeric_classes=True):
    """
    Converts a float target variable to a categorical variable with the specified number of classes

    Parameters
    ----------
    target : pandas.Series
        The target variable to convert
    nb_classes : int
        The number of classes to convert the target variable to
    numeric_classes : bool
        Whether to use numeric classes or not

    Returns
    -------
    pandas.Series
        The converted target variable
    """
    eps = np.finfo(float).eps
    labels = list(range(nb_classes)) if numeric_classes else [f"Q{c}-Q{c + 1}" for c in range(nb_classes)]
    labels_values = np.linspace(0, 1, nb_classes + 1)
    v = np.array(target.quantile(labels_values)).squeeze()
    v[0], v[-1] = v[0] - eps, v[-1] + eps
    y = target.copy()
    for (i, c) in enumerate(labels):
        y[(target.values >= v[i]) & (target.values < v[i + 1])] = c
    return y.astype(np.int32)