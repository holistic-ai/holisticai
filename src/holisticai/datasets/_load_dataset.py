from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from holisticai.datasets._dataloaders import (
    load_adult,
    load_last_fm,
    load_law_school,
    load_student,
    load_us_crime,
)
from holisticai.datasets._dataset import Dataset
from holisticai.datasets._utils import convert_float_to_categorical, get_protected_values


def load_adult_dataset(protected_attribute: Literal["race", "sex"] | None = None):
    data = load_adult()
    output_variable = "class"
    protected_attributes = ["race", "sex"]
    drop_columns = [*protected_attributes, output_variable, "education"]
    df = pd.concat([data["data"], data["target"]], axis=1)
    df = df.dropna().reset_index(drop=True)

    params = {}
    if protected_attribute == "race":
        params["group_a"] = pd.Series(get_protected_values(df, protected_attribute, "White"), name="group_a")
        params["group_b"] = pd.Series(get_protected_values(df, protected_attribute, "Black"), name="group_b")

    if protected_attribute == "sex":
        params["group_a"] = pd.Series(get_protected_values(df, protected_attribute, "Male"), name="group_a")
        params["group_b"] = pd.Series(get_protected_values(df, protected_attribute, "Female"), name="group_b")

    # p_attr = pd.concat([group_a, group_b], axis=1)
    y = df[output_variable].map({"<=50K": 0, ">50K": 1})
    df = df.drop(drop_columns, axis=1)
    df = pd.get_dummies(df, columns=df.columns[df.dtypes == "category"])
    bool_columns = df.columns[df.dtypes == "bool"]
    df[bool_columns] = df[bool_columns].astype(int)
    x = df
    return Dataset(X=x, y=y, **params)


def load_law_school_dataset(protected_attribute: Literal["race1", "gender"] = "race1"):
    bunch = load_law_school()
    protected_attributes = ["race1", "gender"]
    output_variable = "bar"
    drop_columns = ["ugpagt3", "bar", *protected_attributes]
    df = bunch["frame"]
    df = df.dropna()
    params = {}
    if protected_attribute == "race1":
        params["group_a"] = pd.Series(get_protected_values(df, protected_attribute, "white"), name="group_a")
        params["group_b"] = pd.Series(get_protected_values(df, protected_attribute, "non-white"), name="group_b")

    if protected_attribute == "gender":
        params["group_a"] = pd.Series(get_protected_values(df, protected_attribute, "Female"), name="group_a")
        params["group_b"] = pd.Series(get_protected_values(df, protected_attribute, "Male"), name="group_b")

    y = df[output_variable]  # binary label vector
    y = y.map({"FALSE": 0, "TRUE": 1})
    x = df.drop(drop_columns, axis=1)
    return Dataset(X=x, y=y, **params)


def load_student_multiclass_dataset(protected_attribute: Literal["sex", "address"] = "sex"):
    output_column = "G3"
    drop_columns = ["G1", "G2", "G3", "sex", "address", "Mjob", "Fjob"]
    bunch = load_student()
    df = bunch["frame"]

    # we don't want to encode protected attributes
    df = df.dropna()
    y_cat = convert_float_to_categorical(df[output_column], 3)
    p_attr = df[protected_attribute]

    for col in df.select_dtypes(include=["object"]):
        df[col] = pd.factorize(df[col])[0]
    df = df.drop(drop_columns, axis=1)
    df = pd.get_dummies(df, columns=df.columns[df.dtypes == "object"])
    df["target"] = y_cat
    df = df.reset_index(drop=True)

    p_attr = p_attr.reset_index(drop=True)
    x = df.astype(float)
    y = df["target"]
    x = df.drop(columns="target")
    return Dataset(X=x, y=y, p_attr=p_attr)


def load_student_dataset(
    target: Literal["G1", "G2", "G3"] = "G3",
    protected_attribute: Literal["sex", "address", "Mjob", "Fjob", "all"] = "sex",
):
    if protected_attribute not in ["sex", "address", "Mjob", "Fjob", "all"]:
        raise ValueError(f"Unknown protected attribute: {protected_attribute}")

    drop_columns = ["G1", "G2", "G3", "sex", "address", "Mjob", "Fjob"]
    bunch = load_student()

    df = bunch["frame"]
    for col in ["sex", "address", "Mjob", "Fjob"]:
        df[col] = df[col].apply(lambda x: x.replace("'", ""))

    df = df.dropna()
    y = df[target]

    if protected_attribute == "all":
        p_attr = df[["sex", "address", "Mjob", "Fjob"]]
    else:
        p_attr = pd.Series(df[protected_attribute], name="p_attr")
    group_cols = {"p_attr": p_attr}
    if protected_attribute == "sex":
        group_a = pd.Series(df[protected_attribute] == "'F'", name="group_a")
        group_b = ~group_a
        group_cols.update({"group_a": group_a, "group_b": group_b})

    for col in df.select_dtypes(include=["object"]):
        df[col] = pd.factorize(df[col])[0]
    df = df.drop(drop_columns, axis=1)
    df = pd.get_dummies(df, columns=df.columns[df.dtypes == "object"])
    df = df.reset_index(drop=True)

    x = df.astype(float)
    y = y.reset_index(drop=True)
    return Dataset(X=x, y=y, **group_cols)


def load_lastfm_dataset():
    """
    Processes the lastfm dataset and returns the data, output variable, protected group A and protected group B as numerical arrays

    Parameters
    ----------
    size : str
        The size of the dataset to return. Either 'small' or 'large'

    Returns
    -------
    data_matrix : np.ndarray
        The numerical pivot array
    p_attr : np.ndarray
        The protected attribute
    """
    bunch = load_last_fm()
    df = bunch["frame"]
    protected_attribute = "sex"
    user_column = "user"
    item_column = "artist"

    from holisticai.utils import recommender_formatter

    random_state = np.random.RandomState(42)
    df["score"] = random_state.randint(1, 5, len(df))
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
    df_pivot = df_pivot.fillna(0)
    return Dataset(data_pivot=df_pivot, p_attr=pd.Series(p_attr))


def load_us_crime_dataset():
    """
    Processes the US crime dataset and returns the data, output variable, protected group A and \
    protected group B as numerical arrays or as dataframe if needed

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
    min_nonan_values = 1000
    data = load_us_crime()
    protected_attribute = "racePctWhite"
    output_variable = "ViolentCrimesPerPop"
    df = pd.concat([data["data"], data["target"]], axis=1)
    df = df.iloc[:, [i for i, n in enumerate(df.isna().sum(axis=0).T.values) if n < min_nonan_values]]
    df = df.dropna()
    threshold = 0.5
    group_a = pd.Series(df[protected_attribute] > threshold, name="group_a")
    group_b = pd.Series(~group_a, name="group_b")
    y = df[output_variable]
    remove_columns = [protected_attribute, output_variable]
    x = df.drop(columns=remove_columns)
    numeric_features = x.select_dtypes(include=[np.number]).columns
    return Dataset(X=x[numeric_features], y=y, group_a=group_a, group_b=group_b)


def load_us_crime_multiclass_dataset():
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
    min_nonan_values = 1000
    data = load_us_crime()
    protected_attribute = "racePctWhite"
    output_column = "ViolentCrimesPerPop"
    df = pd.concat([data["data"], data["target"]], axis=1)
    df = df.iloc[:, [i for i, n in enumerate(df.isna().sum(axis=0).T.values) if n < min_nonan_values]]
    df = df.dropna()
    threshold = 0.5
    group_a = pd.Series(df[protected_attribute] > threshold, name="group_a")
    group_b = pd.Series(~group_a, name="group_b")
    p_attr = pd.Series([1 if i else 0 for i in group_a])
    y_cat = pd.Series(convert_float_to_categorical(df[output_column], 3))
    remove_columns = [protected_attribute, output_column]
    x = df.drop(columns=remove_columns)
    numeric_features = x.select_dtypes(include=[np.number]).columns
    return Dataset(X=x[numeric_features], y=y_cat, group_a=group_a, group_b=group_b, p_attr=p_attr)


def load_clinical_records_dataset():
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
    df = df.dropna().reset_index(drop=True)
    group_a = pd.Series(df[protected_attribute] == 0, name="group_a")
    group_b = pd.Series(df[protected_attribute] == 1, name="group_b")

    y = df[output_variable]
    x = df.drop(columns=drop_columns)
    return Dataset(X=x, y=y, group_a=group_a, group_b=group_b)


ProcessedDatasets = Literal[
    "adult",
    "law_school",
    "student_multiclass",
    "student",
    "lastfm",
    "us_crime",
    "us_crime_multiclass",
    "clinical_records",
]


def load_dataset(dataset_name: ProcessedDatasets, **kargs) -> Dataset:
    """
    Load a specific dataset based on the given dataset name.

    Parameters
    ----------
    dataset_name: ProcessedDatasets
        The name of the dataset to load. The list of supported datasets are here: :ref:`processed_datasets`.
    kargs:
        Additional keyword arguments to pass to the dataset loading functions.

    Returns
    -------
    Dataset: The loaded dataset.

    Raises
    ------
    NotImplementedError:
        If the specified dataset name is not supported.
    """
    if dataset_name == "adult":
        return load_adult_dataset(**kargs)
    if dataset_name == "law_school":
        return load_law_school_dataset(**kargs)
    if dataset_name == "student_multiclass":
        return load_student_multiclass_dataset()
    if dataset_name == "student":
        return load_student_dataset(**kargs)
    if dataset_name == "lastfm":
        return load_lastfm_dataset()
    if dataset_name == "us_crime":
        return load_us_crime_dataset()
    if dataset_name == "us_crime_multiclass":
        return load_us_crime_multiclass_dataset()
    if dataset_name == "clinical_records":
        return load_clinical_records_dataset()
    raise NotImplementedError
