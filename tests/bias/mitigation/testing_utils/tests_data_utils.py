import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from holisticai.datasets import load_adult, load_us_crime


def load_preprocessed_adult(short_version=True):
    dataset = load_adult()
    df = pd.concat([dataset["data"], dataset["target"]], axis=1)
    if not short_version:
        df = df.sample(n=500)

    protected_variables = ["sex", "race"]
    output_variable = ["class"]
    favorable_label = 1
    unfavorable_label = 0

    y = df[output_variable].replace(
        {">50K": favorable_label, "<=50K": unfavorable_label}
    )
    x = pd.get_dummies(df.drop(protected_variables + output_variable, axis=1))

    group = ["sex"]
    group_a = df[group] == "Female"
    group_b = df[group] == "Male"
    data = [x, y, group_a, group_b]

    dataset = train_test_split(*data, test_size=0.2, shuffle=True)
    train_data = dataset[::2]
    test_data = dataset[1::2]
    return train_data, test_data


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


def convert_float_to_categorical(target, nb_classes, numeric_classes=True):
    """
    Converts a float target variable to a categorical variable with the specified number of classes

    Parameters
    ----------
    target : pandas.Series
        The target variable to convert
    nb_classes : int
        The number of classes to convert the target variable to
    numeric_classes : bool
        Whether to use numeric classes or not

    Returns
    -------
    pandas.Series
        The converted target variable
    """
    eps = np.finfo(float).eps
    if numeric_classes:
        labels = list(range(nb_classes))
    else:
        labels = [f"Q{c}-Q{c+1}" for c in range(nb_classes)]
    labels_values = np.linspace(0, 1, nb_classes + 1)
    v = np.array(target.quantile(labels_values)).squeeze()
    v[0], v[-1] = v[0] - eps, v[-1] + eps
    y = target.copy()
    for (i, c) in enumerate(labels):
        y[(target.values >= v[i]) & (target.values < v[i + 1])] = c
    return y.astype(np.int32)


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


def sample_data(
    df,
    protected_attribute,
    output_variable,
    protected_values,
    output_values,
    num_samples,
):
    df_list = []
    for p in protected_values:
        for o in output_values:
            df_list.append(
                df[(df[protected_attribute] == p) & (df[output_variable] == o)]
                .sample(num_samples)
                .reset_index(drop=True)
            )
    return pd.concat(df_list, axis=0)


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


def sample_categorical(df, group_a, group_b, output_variable):
    from sklearn.preprocessing import StandardScaler

    y = convert_float_to_categorical(df[output_variable], 3)
    scalar = StandardScaler()
    df = scalar.fit_transform(df)
    X = df[:, :-1]
    data = []
    for m in [X, y, group_a, group_b]:
        x = pd.DataFrame(m.copy())
        x = pd.concat(
            [
                x[(group_a == 1) & (y == 0)].iloc[:2],
                x[(group_a == 1) & (y == 1)].iloc[:2],
                x[(group_a == 1) & (y == 2)].iloc[:2],
                x[(group_b == 1) & (y == 0)].iloc[:2],
                x[(group_b == 1) & (y == 1)].iloc[:2],
                x[(group_b == 1) & (y == 2)].iloc[:2],
            ],
            axis=0,
        ).reset_index(drop=True)
        data.append(x)
    df = pd.concat(data[:2], axis=1)
    return df, data[2], data[3]


def post_process_recommender(df):
    df = df.fillna(0)
    data_matrix = df.to_numpy()
    return data_matrix


def preprocess_lastfm_dataset(df, protected_attribute, user_column, item_column):
    """
    Performs the pre-processing step of the data.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to pre-process
    protected_attribute : str
        The name of the protected attribute
    user_column : str
        The name of the user column
    item_column : str
        The name of the item column

    Returns
    -------
    df_pivot : pandas.DataFrame
        The pre-processed dataframe
    p_attr : np.ndarray
        The protected attribute
    """
    from holisticai.utils import recommender_formatter

    df["score"] = np.random.randint(1, 5, len(df))
    df[protected_attribute] = df[protected_attribute] == "m"
    df = df.drop_duplicates()
    df_pivot, p_attr = recommender_formatter(
        df,
        users_col=user_column,
        groups_col=protected_attribute,
        items_col=item_column,
        scores_col="score",
        aggfunc="mean",
    )
    return df_pivot, p_attr


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
    return data, data


def load_preprocessed_us_crime(nb_classes=None):
    from sklearn.preprocessing import StandardScaler

    dataset = load_us_crime()

    df = pd.concat([dataset["data"], dataset["target"]], axis=1)
    df_clean = df.iloc[
        :, [i for i, n in enumerate(df.isna().sum(axis=0).T.values) if n < 1000]
    ]
    df_clean = df_clean.dropna()

    group_a = df_clean["racePctWhite"].apply(lambda x: x > 0.5)
    group_b = 1 - group_a
    xor_groups = group_a ^ group_b

    # Remove sensitive groups from dataset
    cols = [
        c
        for c in df_clean.columns
        if (not c.startswith("race")) and (not c.startswith("age"))
    ]
    df_clean = df_clean[cols].iloc[:, 3:]
    df_clean = df_clean[xor_groups]
    group_a = group_a[xor_groups]
    group_b = group_b[xor_groups]

    scalar = StandardScaler()
    df_t = scalar.fit_transform(df_clean)
    X = df_t[:, :-1]

    if nb_classes is None:
        y = df_t[:, -1]
    else:
        y = convert_float_to_categorical(df_clean.iloc[:, -1], nb_classes=nb_classes)

    data = X, y, group_a, group_b
    datasets = train_test_split(*data, test_size=0.2)
    train_data = datasets[::2]
    test_data = datasets[1::2]
    return train_data, test_data
