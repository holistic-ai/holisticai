import pandas as pd

from holisticai.datasets import load_adult

from .tests_data_utils import (
    get_protected_values,
    post_process_dataframe,
    post_process_dataset,
    remove_nans,
)


def preprocess_adult_dataset(df, protected_attribute, output_variable, drop_columns):
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
    df = df.dropna()
    group_a = get_protected_values(df, protected_attribute, "Female")
    group_b = get_protected_values(df, protected_attribute, "Male")
    unique_values = df[output_variable].unique()
    output = df[output_variable].map({unique_values[0]: 0, unique_values[1]: 1})
    df = df.drop(drop_columns, axis=1)
    df = pd.get_dummies(df, columns=df.columns[df.dtypes == "category"])
    df[output_variable] = output
    return post_process_dataframe(df, group_a, group_b)


def sample_adult(df, protected_attribute="sex", output_variable="class"):
    df = pd.concat(
        [
            df[(df[protected_attribute] == "Male") & (df[output_variable] == ">50K")]
            .sample(50)
            .reset_index(drop=True),
            df[(df[protected_attribute] == "Male") & (df[output_variable] == "<=50K")]
            .sample(50)
            .reset_index(drop=True),
            df[(df[protected_attribute] == "Female") & (df[output_variable] == ">50K")]
            .sample(50)
            .reset_index(drop=True),
            df[(df[protected_attribute] == "Female") & (df[output_variable] == "<=50K")]
            .sample(50)
            .reset_index(drop=True),
        ],
        axis=0,
    )
    return df


def process_binary_dataset(size="small"):
    """
    Processes the adult dataset and returns the data, output variable, protected group A and protected group B as numerical arrays

    Parameters
    ----------
    size : str
        The size of the dataset to return. Either 'small' or 'large'

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    data = load_adult()
    protected_attribute = "sex"
    output_variable = "class"
    drop_columns = ["education", "race", "sex", "class"]
    df = pd.concat([data["data"], data["target"]], axis=1)
    df = remove_nans(df)
    if size == "small":
        df = sample_adult(df)
    df, group_a, group_b = preprocess_adult_dataset(
        df, protected_attribute, output_variable, drop_columns
    )
    return post_process_dataset(df, output_variable, group_a, group_b)
