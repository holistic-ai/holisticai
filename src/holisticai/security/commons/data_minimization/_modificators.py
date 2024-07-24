from typing import Literal

import numpy as np
import pandas as pd

ModifierType = Literal["Average", "Permutation"]


class ModifierHandler:
    def __init__(self, methods: ModifierType):
        self.methods = methods

    def __call__(self, x, indexes) -> pd.DataFrame:
        results = {}
        for method in self.methods:
            new_x, updated_indexes = self.apply_method(method, x, indexes)
            results[method] = {"x": new_x, "updated_features": updated_indexes}
        return results

    def apply_method(self, method_name: ModifierType, x, indexes) -> pd.DataFrame:
        if method_name == "Average":
            return replace_data_with_average(x, indexes)
        if method_name == "Permutation":
            return replace_data_with_permutation(x, indexes)
        raise NotImplementedError(f"Method {method_name} not implemented")


def replace_data_with_average(x, important_feature_names):
    """
    Replaces the values of columns that are not in `important_feature_names` with the mean value of each column.

    Parameters
    ----------
    x : pandas.DataFrame
        The input DataFrame containing the data to be modified.
    important_feature_names : list
        A list of column names that should not be modified.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the modified values.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    >>> important_features = ["A", "C"]
    >>> replace_data_with_average(data, important_features)
       A  B  C
    0  1  6  7
    1  2  6  8
    2  3  6  9
    """
    feature_names = x.columns
    no_important_feature_names = [f for f in feature_names if f not in important_feature_names]

    xdf = x.copy()
    average = x[no_important_feature_names].select_dtypes(include=["number"]).mean()

    xdf[average.index] = average

    return xdf, list(average.index)


def replace_data_with_permutation(x: pd.DataFrame, important_feature_names: list) -> pd.DataFrame:
    """
    Replace non-important column values with random samples chosen from the same column.

    Parameters
    ----------
    x : pd.DataFrame
        The input DataFrame.
    important_feature_names : list
        List of column names that should not be replaced.

    Returns
    -------
    pd.DataFrame
        DataFrame with non-important columns replaced by random samples.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    >>> important_features = ["A", "C"]
    >>> replace_data_with_permutation(df, important_features)
       A  B  C
    0  1  5  7
    1  2  6  8
    2  3  4  9
    """
    feature_names = x.columns
    no_important_feature_names = [f for f in feature_names if f not in important_feature_names]

    xdf = x.copy()

    for yc in no_important_feature_names:
        unique_values = np.unique(xdf.loc[:, yc])
        xdf.loc[:, yc] = np.random.choice(unique_values, len(xdf))

    return xdf, no_important_feature_names
