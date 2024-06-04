import numpy as np
import pandas as pd

from holisticai.datasets import load_us_crime

from .data_utils import post_process_dataframe, post_process_dataset, remove_nans


def preprocess_us_crime_dataset(df, protected_attribute, threshold=0.5):
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


def process_regression_dataset(size="small", return_df=False):
    """
    Processes the US crime dataset and returns the data, output variable, protected group A and protected group B as numerical arrays or as dataframe if needed

    Parameters
    ----------
    size : str
        The size of the dataset to return. Either 'small' or 'large'
    return_df : bool
        Whether to return the data as dataframe or not

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    data = load_us_crime()
    protected_attribute = "racePctWhite"
    output_variable = "ViolentCrimesPerPop"
    df = pd.concat([data["data"], data["target"]], axis=1)
    df = remove_nans(df)
    df, group_a, group_b = preprocess_us_crime_dataset(
        df, protected_attribute=protected_attribute
    )
    if size == "small":
        a_index = list(np.where(group_a == 1)[0][::5])
        b_index = list(np.where(group_b == 1)[0][::5])
        df = df.iloc[a_index + b_index]
        group_a = group_a.iloc[a_index + b_index]
        group_b = group_b.iloc[a_index + b_index]
    if return_df:
        return post_process_dataframe(df, group_a, group_b)
    df, group_a, group_b = post_process_dataframe(df, group_a, group_b)
    y = df[output_variable]
    X = df.drop(columns=output_variable)
    return post_process_dataset(df, output_variable, group_a, group_b)
