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
    eps = 10 * np.finfo(float).eps
    labels = list(range(nb_classes)) if numeric_classes else [f"Q{c}-Q{c + 1}" for c in range(nb_classes)]
    labels_values = np.linspace(0, 1, nb_classes + 1)
    v = np.array(target.quantile(labels_values)).squeeze()
    v[0], v[-1] = v[0] - eps, v[-1] + eps
    y = target.copy()
    for i, c in enumerate(labels):
        y[(target.values >= v[i]) & (target.values < v[i + 1])] = c
    return y.astype("category")


def get_protected_values(df, protected_attribute, protected_value):
    """
    Returns a boolean array with True for the protected group and False for the unprotected group

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the protected attribute
    protected_attribute : str
        The name of the protected attribute
    protected_value : str
        The value of the protected attribute for the protected group

    Returns
    -------
    np.ndarray
        A boolean array with True for the protected group and False for the unprotected group
    """
    return df[protected_attribute] == protected_value
