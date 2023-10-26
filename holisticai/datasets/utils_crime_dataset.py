import pandas as pd

from ._dataloaders import load_us_crime
from .dataset_processing_utils import post_process_dataset, remove_nans


def __preprocess_us_crime_dataset(df, protected_attribute, threshold=0.5):
    """
    Pre-processes the US crime dataset by converting the output variable to a binary variable, dropping unnecessary columns, converting categorical columns to one-hot encoded columns and converting the output variable to a binary variable

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to pre-process
    protected_attribute : str
        The name of the protected attribute
    threshold : float
        The threshold to use to split the protected attribute into two groups

    Returns
    -------
    df : pandas.DataFrame
        The pre-processed dataframe
    group_a : pandas.DataFrame
        The dataframe containing the protected group A
    group_b : pandas.DataFrame
        The dataframe containing the protected group B
    """
    group_a = df[protected_attribute] > threshold
    group_b = ~group_a
    xor_groups = group_a ^ group_b
    cols = [c for c in df.columns if not (c.startswith("race") or c.startswith("age"))]
    df = df[cols].iloc[:, 3:].loc[xor_groups]
    group_a, group_b = group_a[xor_groups], group_b[xor_groups]
    return df, group_a, group_b


def process_crime_dataset(as_array=False):
    """
    Processes the US crime dataset with some fixed parameters and returns the data and protected groups. If as_array is True, returns the data as numpy arrays. If as_array is False, returns the data as pandas dataframes

    Parameters
    ----------
    as_array : bool
        If True, returns the data as numpy arrays. If False, returns the data as pandas dataframes

    Returns
    -------
    tuple
        When as_array is True, returns a tuple with four numpy arrays containing the data, output variable, protected group A and protected group B. When as_array is False, returns a tuple with three pandas dataframes containing the data, protected group A and protected group B
    """

    data = load_us_crime()
    protected_attribute = "racePctWhite"
    output_variable = "ViolentCrimesPerPop"
    df = pd.concat([data["data"], data["target"]], axis=1)
    df = remove_nans(df)
    df, group_a, group_b = __preprocess_us_crime_dataset(
        df, protected_attribute=protected_attribute
    )
    if as_array:
        return post_process_dataset(df, output_variable, group_a, group_b)
    return df, group_a, group_b
