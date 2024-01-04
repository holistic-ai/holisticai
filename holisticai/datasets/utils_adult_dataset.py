import pandas as pd

from ._dataloaders import load_adult
from .dataset_processing_utils import (
    get_protected_values,
    post_process_dataframe,
    post_process_dataset,
    remove_nans,
)


def __preprocess_adult_dataset(df, protected_attribute, output_variable, drop_columns):
    """
    Pre-processes the adult dataset by converting the output variable to a binary variable, dropping unnecessary columns, converting categorical columns to one-hot encoded columns and converting the output variable to a binary variable

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
    df : pandas.DataFrame
        The pre-processed dataframe
    group_a : pandas.DataFrame
        The dataframe containing the protected group A
    group_b : pandas.DataFrame
        The dataframe containing the protected group B
    """
    group_a = get_protected_values(df, protected_attribute, "Female")
    group_b = get_protected_values(df, protected_attribute, "Male")
    unique_values = df[output_variable].unique()
    output = df[output_variable].map({unique_values[0]: 0, unique_values[1]: 1})
    df = df.drop(drop_columns, axis=1)
    df = pd.get_dummies(df, columns=df.columns[df.dtypes == "category"])
    df[output_variable] = output
    return post_process_dataframe(df, group_a, group_b)


def process_adult_dataset(as_array=False):
    """
    Processes the adult dataset with some fixed parameters and returns the data and protected groups. If as_array is True, returns the data as numpy arrays. If as_array is False, returns the data as pandas dataframes

    Parameters
    ----------
    as_array : bool
        If True, returns the data as numpy arrays. If False, returns the data as pandas dataframes

    Returns
    -------
    tuple
        When as_array is True, returns a tuple with four numpy arrays containing the data, output variable, protected group A and protected group B. When as_array is False, returns a tuple with three pandas dataframes containing the data, protected group A and protected group B
    """
    data = load_adult()
    protected_attribute = "sex"
    output_variable = "class"
    drop_columns = ["education", "race", "sex", "class"]
    df = pd.concat([data["data"], data["target"]], axis=1)
    df = remove_nans(df)
    df, group_a, group_b = __preprocess_adult_dataset(
        df, protected_attribute, output_variable, drop_columns
    )
    if as_array:
        return post_process_dataset(df, output_variable, group_a, group_b)
    return df, group_a, group_b
