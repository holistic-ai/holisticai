import numpy as np
import pandas as pd

from .data_utils import post_process_dataframe, post_process_dataset
from .regression_dataset import process_regression_dataset


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
    if numeric_classes:
        labels = list(range(nb_classes))
    else:
        labels = [f"Q{c}-Q{c+1}" for c in range(nb_classes)]
    labels_values = np.linspace(0, 1, nb_classes + 1)
    v = np.array(target.quantile(labels_values)).squeeze()
    v[0], v[-1] = v[0] - eps, v[-1] + eps
    y = target.copy()
    for (i, c) in enumerate(labels):
        y[(target.values >= v[i]) & (target.values < v[i + 1])] = c
    return y.astype(np.int32)


def sample_categorical(df, group_a, group_b, output_variable):
    from sklearn.preprocessing import StandardScaler

    y = convert_float_to_categorical(df[output_variable], 3)
    scalar = StandardScaler()
    df = scalar.fit_transform(df)
    X = df[:, :-1]
    data = []
    for m in [X, y, group_a, group_b]:
        x = pd.DataFrame(m.copy())
        x = pd.concat(
            [
                x[(group_a == 1) & (y == 0)].iloc[:2],
                x[(group_a == 1) & (y == 1)].iloc[:2],
                x[(group_a == 1) & (y == 2)].iloc[:2],
                x[(group_b == 1) & (y == 0)].iloc[:2],
                x[(group_b == 1) & (y == 1)].iloc[:2],
                x[(group_b == 1) & (y == 2)].iloc[:2],
            ],
            axis=0,
        ).reset_index(drop=True)
        data.append(x)
    df = pd.concat(data[:2], axis=1)
    return df, data[2], data[3]


def process_multiclass_dataset(size="small"):
    """
    Processes the US crime dataset as a multiclass dataset and returns the data, output variable, protected group A and protected group B as numerical arrays

    Parameters
    ----------
    size : str
        The size of the dataset to return. Either 'small' or 'large'

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    output_variable = "ViolentCrimesPerPop"
    df, group_a, group_b = process_regression_dataset("large", return_df=True)
    if size == "small":
        df, group_a, group_b = sample_categorical(df, group_a, group_b, output_variable)
        df, group_a, group_b = post_process_dataframe(df, group_a, group_b)
        return post_process_dataset(df, output_variable, group_a, group_b)
    print(df.shape)
    df[output_variable] = convert_float_to_categorical(df[output_variable], 5)
    df, group_a, group_b = post_process_dataframe(df, group_a, group_b)
    return post_process_dataset(df, output_variable, group_a, group_b)
