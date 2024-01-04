import pandas as pd

from ._dataloaders import load_heart
from .dataset_processing_utils import (
    get_protected_values,
    post_process_dataframe,
    post_process_dataset,
    remove_nans,
)


def __preprocess_heart_dataset(df, protected_attribute, output_variable, drop_columns):
    """
    Pre-processes the heart dataset and returns the data, output variable, protected group A and protected group B as numerical arrays

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to pre-process
    protected_attribute : str
        The name of the protected attribute
    output_variable : str
        The name of the output variable
    drop_columns : list
        The list of columns to drop

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    group_a = get_protected_values(df, protected_attribute, 0)
    group_b = get_protected_values(df, protected_attribute, 1)
    output = df[output_variable]
    df = df.drop(columns=drop_columns)
    df[output_variable] = output
    return post_process_dataframe(df, group_a, group_b)


def process_heart_dataset(as_array=False):
    """
    Processes the heart dataset and returns the data, output variable, protected group A and protected group B as numerical arrays

    Parameters
    ----------
    as_array : bool
        If True, returns the data as numpy arrays. If False, returns the data as pandas dataframes

    Returns
    -------
    tuple
        When as_array is True, returns a tuple with four numpy arrays containing the data, output variable, protected group A and protected group B. When as_array is False, returns a tuple with three pandas dataframes containing the data, protected group A and protected group B
    """
    data = load_heart()
    protected_attribute = "sex"
    output_variable = "DEATH_EVENT"
    drop_columns = ["age", "sex", "DEATH_EVENT"]
    df = pd.concat([data["data"], data["target"]], axis=1)
    df = remove_nans(df)
    df, group_a, group_b = __preprocess_heart_dataset(
        df, protected_attribute, output_variable, drop_columns
    )
    if as_array:
        return post_process_dataset(df, output_variable, group_a, group_b)
    return df, group_a, group_b
