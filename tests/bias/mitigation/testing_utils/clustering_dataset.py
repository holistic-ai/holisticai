import pandas as pd

from .data_utils import (
    get_protected_values,
    post_process_dataframe,
    post_process_dataset,
    remove_nans,
)


def sample_heart(df, protected_attribute="sex", output_variable="DEATH_EVENT"):
    df = pd.concat(
        [
            df[(df[protected_attribute] == 1) & (df[output_variable] == 1)]
            .sample(10)
            .reset_index(drop=True),
            df[(df[protected_attribute] == 1) & (df[output_variable] == 0)]
            .sample(10)
            .reset_index(drop=True),
            df[(df[protected_attribute] == 0) & (df[output_variable] == 1)]
            .sample(10)
            .reset_index(drop=True),
            df[(df[protected_attribute] == 0) & (df[output_variable] == 0)]
            .sample(10)
            .reset_index(drop=True),
        ],
        axis=0,
    )
    return df


def preprocess_heart_dataset(df, protected_attribute, output_variable, drop_columns):
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


def process_clustering_dataset(size="small"):
    """
    Processes the heart dataset and returns the data, output variable, protected group A and protected group B as numerical arrays

    Parameters
    ----------
    size : str
        The size of the dataset to return. Either 'small' or 'large'

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
    )
    protected_attribute = "sex"
    output_variable = "DEATH_EVENT"
    drop_columns = ["age", "sex", "DEATH_EVENT"]
    df = remove_nans(df)
    if size == "small":
        df = sample_heart(df)
    df, group_a, group_b = preprocess_heart_dataset(
        df, protected_attribute, output_variable, drop_columns
    )
    y = df[output_variable]
    X = df.drop(columns=output_variable)
    return post_process_dataset(df, output_variable, group_a, group_b)
