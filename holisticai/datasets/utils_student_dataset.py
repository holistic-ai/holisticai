import numpy as np
import pandas as pd

from ._dataloaders import load_student


def preprocess_student_dataset(df, protected_attribute, output_variable, drop_columns):
    """
    Pre-processes the student dataset by converting the output variable to a categorical variable, dropping unnecessary columns and converting categorical columns to numerical encoded columns.

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
    p_attr : pandas.DataFrame
        The dataframe containing the protected attribute
    """
    df = df.dropna()
    y = df[output_variable].to_numpy()
    buckets = np.array([8, 11, 14])
    y_cat = (y.reshape(-1, 1) > buckets.reshape(1, -1)).sum(axis=1)
    p_attr = df[protected_attribute]
    for col in df.select_dtypes(include=["object"]):
        df[col] = pd.factorize(df[col])[0]
    df = df.drop(drop_columns, axis=1)
    df = pd.get_dummies(df, columns=df.columns[df.dtypes == "object"])
    df["target"] = y_cat
    df = df.reset_index(drop=True)
    p_attr = p_attr.reset_index(drop=True)
    df = df.astype(float)
    return df, p_attr


def process_student_dataset(as_array=False):
    """
    Processes the student dataset with some fixed parameters and returns the data and protected groups. If as_array is True, returns the data as numpy arrays. If as_array is False, returns the data as pandas dataframes

    Parameters
    ----------
    as_array : bool
        If True, returns the data as numpy arrays. If False, returns the data as pandas dataframes

    Returns
    -------
    tuple
        When as_array is True, returns a tuple with three numpy arrays containing the data, output variable and protected attribute. When as_array is False, returns a tuple with two pandas dataframes containing the data and the protected attribute
    """
    protected_attribute = "Mjob"
    output_variable = "G3"
    drop_columns = ["G1", "G2", "G3", "Mjob", "age", "Fjob"]
    bunch = load_student()
    df = bunch["frame"]
    df, p_attr = preprocess_student_dataset(
        df, protected_attribute, output_variable, drop_columns
    )
    if as_array:
        y = df["target"]
        X = df.drop(columns="target")
        data = [X.values, y.values.ravel(), p_attr.values.ravel()]
        return data
    return df, p_attr
