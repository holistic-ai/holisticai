def post_process_recommender(df):
    df = df.fillna(0)
    data_matrix = df.to_numpy()
    return data_matrix


def get_protected_values(df, protected_attribute, protected_value):
    """
    Returns a boolean array with True for the protected group and False for the unprotected group

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the protected attribute
    protected_attribute : str
        The name of the protected attribute
    protected_value : str
        The value of the protected attribute for the protected group

    Returns
    -------
    np.ndarray
        A boolean array with True for the protected group and False for the unprotected group
    """
    group = df[protected_attribute] == protected_value
    return group


def remove_nans(df):
    """
    Removes columns with more than 1000 NaNs and rows with NaNs

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to remove NaNs from

    Returns
    -------
    pandas.DataFrame
        The dataframe without NaNs
    """
    df = df.iloc[
        :, [i for i, n in enumerate(df.isna().sum(axis=0).T.values) if n < 1000]
    ]
    df = df.dropna()
    return df


def post_process_dataset(df, output_variable, group_a, group_b):
    """
    Post-processes a dataset by returning the data, output variable, protected group A and protected group B as numerical arrays

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to post-process
    output_variable : str
        The name of the output variable
    group_a : pandas.DataFrame
        The dataframe containing the protected group A
    group_b : pandas.DataFrame
        The dataframe containing the protected group B

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    y = df[output_variable]
    X = df.drop(columns=output_variable)
    data = [
        X.values,
        y.values.ravel(),
        group_a.values.ravel(),
        group_b.values.ravel(),
    ]
    return data


def post_process_dataframe(df, group_a, group_b):
    """
    Post-processes a dataframe by resetting the index, converting the dataframe to float and resetting the index of the protected groups

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to post-process
    group_a : pandas.DataFrame
        The dataframe containing the protected group A
    group_b : pandas.DataFrame
        The dataframe containing the protected group B

    Returns
    -------
    df : pandas.DataFrame
        The post-processed dataframe
    group_a : pandas.DataFrame
        The post-processed dataframe containing the protected group A
    group_b : pandas.DataFrame
        The post-processed dataframe containing the protected group B
    """
    df = df.reset_index(drop=True)
    group_a = group_a.reset_index(drop=True)
    group_b = group_b.reset_index(drop=True)
    df = df.astype(float)
    return df, group_a, group_b
