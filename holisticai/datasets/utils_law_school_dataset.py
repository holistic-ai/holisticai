from ._dataloaders import load_law_school
from .dataset_processing_utils import (
    get_protected_values,
    post_process_dataframe,
    post_process_dataset,
)


def preprocess_law_school_dataset(
    df, protected_attribute, output_variable, drop_columns
):
    """
    Pre-processes the law school dataset by converting the output variable to a binary variable, dropping unnecessary columns and converting categorical columns to one-hot encoded columns.

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
    df = df.dropna()
    group_a = get_protected_values(df, protected_attribute, "white")
    group_b = get_protected_values(df, protected_attribute, "non-white")
    y = df[output_variable]  # binary label vector
    y = y.replace({"FALSE": 0, "TRUE": 1})
    df = df.drop(drop_columns, axis=1)
    df["target"] = y
    return post_process_dataframe(df, group_a, group_b)


def process_law_school_dataset(as_array=False):
    """
    Processes the law school dataset with some fixed parameters and returns the data and protected groups. If as_array is True, returns the data as numpy arrays. If as_array is False, returns the data as pandas dataframes
    For convenience, the output variable is renamed to 'target'.

    Parameters
    ----------
    as_array : bool
        If True, returns the data as numpy arrays. If False, returns the data as pandas dataframes

    Returns
    -------
    tuple
        When as_array is True, returns a tuple with four numpy arrays containing the data, output variable, protected group A and protected group B. When as_array is False, returns a tuple with three pandas dataframes containing the data, protected group A and protected group B
    """
    protected_attribute = "race1"
    output_variable = "bar"
    drop_columns = ["ugpagt3", "race1", "gender", "bar"]
    bunch = load_law_school()
    df = bunch["frame"]
    df, group_a, group_b = preprocess_law_school_dataset(
        df, protected_attribute, output_variable, drop_columns
    )
    if as_array:
        return post_process_dataset(df, "target", group_a, group_b)
    return df, group_a, group_b
