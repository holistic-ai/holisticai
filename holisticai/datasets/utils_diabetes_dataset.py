import numpy as np
import pandas as pd

from ._dataloaders import load_diabetes
from .dataset_processing_utils import (
    get_protected_values,
    post_process_dataframe,
    post_process_dataset,
    remove_nans,
)


def __preprocess_diabetes_dataset(
    df, protected_attribute, output_variable, drop_columns
):
    """
    Pre-processes the diabetes dataset by converting the output variable to a binary variable, dropping unnecessary columns, converting categorical columns to one-hot encoded columns and converting the output variable to a binary variable

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
    group_a = get_protected_values(df, protected_attribute, "white")
    group_b = get_protected_values(df, protected_attribute, "non-white")
    unique_values = df[output_variable].unique()
    output = df[output_variable].map({unique_values[0]: 0, unique_values[1]: 1})
    df = df.drop(drop_columns, axis=1)
    df = pd.get_dummies(df, columns=df.columns[df.dtypes == "category"])
    df[output_variable] = output
    return post_process_dataframe(df, group_a, group_b)


def process_diabetes_dataset(as_array=False):
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
    data = load_diabetes()
    protected_attribute = "race"
    output_variable = "target"
    drop_columns = [
        "race",
        "sex",
        "age",
        "weight",
        "payer_code",
        "diag_1",
        "diag_2",
        "diag_3",
        "encounter_id",
        "patient_nbr",
        "medical_specialty",
        "target",
    ]
    df = pd.concat([data["data"], data["target"]], axis=1)
    df.rename(columns={"readmitted": output_variable, "gender": "sex"}, inplace=True)
    df["race"] = np.where(df["race"] == "Caucasian", "white", "non-white")
    df[output_variable] = np.where(df[output_variable] == "NO", 0, 1)
    df = remove_nans(df)
    df, group_a, group_b = __preprocess_diabetes_dataset(
        df, protected_attribute, output_variable, drop_columns
    )
    if as_array:
        return post_process_dataset(df, output_variable, group_a, group_b)
    return df, group_a, group_b
