from __future__ import annotations

from typing import Literal, Union

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


def create_preprocessor(X, numerical_transform: bool = True, categorical_transform: bool = True):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorical_features = X.select_dtypes(include=["category"]).columns
    numerical_fatures = X.select_dtypes(exclude=["category"]).columns

    # Create transformers for numerical and categorical features
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    transformers = []
    if numerical_transform:
        transformers.append(("num", numeric_transformer, numerical_fatures))
    if categorical_transform:
        transformers.append(("cat", categorical_transformer, categorical_features))

    # Combine transformers into a preprocessor using ColumnTransformer
    return ColumnTransformer(transformers=transformers)


def load_adult_dataset(protected_attribute: Union[Literal["race", "sex"], None] = None, preprocessed: bool = True):
    sensitive_attribute = ["race", "sex"]
    feature_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
    ]
    column_types = {
        "age": "int64",
        "fnlwgt": "object",
        "workclass": "category",
        "education": "category",
        "marital-status": "category",
        "occupation": "category",
        "relationship": "category",
        "capital-gain": "int64",
        "capital-loss": "int64",
        "hours-per-week": "int64",
        "native-country": "category",
    }
    output_variable = "class"

    data = load_adult()
    df = pd.concat([data["data"], data["target"]], axis=1)
    df = df.dropna().reset_index(drop=True)
    p_attrs = df[sensitive_attribute]
    if preprocessed:
        if protected_attribute is not None:
            if protected_attribute == "race":
                ga_label = "White"
                gb_label = "Black"
                group_a = pd.Series(get_protected_values(df, protected_attribute, ga_label), name="group_a")
                group_b = pd.Series(get_protected_values(df, protected_attribute, gb_label), name="group_b")

            elif protected_attribute == "sex":
                ga_label = "Male"
                gb_label = "Female"
                group_a = pd.Series(get_protected_values(df, protected_attribute, ga_label), name="group_a")
                group_b = pd.Series(get_protected_values(df, protected_attribute, gb_label), name="group_b")
            else:
                raise ValueError("The protected attribute must be: race or sex")

        y = df[output_variable].map({"<=50K": 0, ">50K": 1}).astype("category")
        xt = df[feature_names]
        categorical_features = xt.select_dtypes(include=["category"]).columns
        xt = pd.get_dummies(xt, columns=categorical_features).astype(np.float64)
        if protected_attribute is not None:
            metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
            return Dataset(X=xt, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
        return Dataset(X=xt, y=y, p_attrs=p_attrs)

    x = df[feature_names].astype(column_types)
    y = df[output_variable]
    return Dataset(X=x, y=y, p_attrs=p_attrs)


def load_law_school_dataset(
    protected_attribute: Union[Literal["race", "gender"], None] = None, preprocessed: bool = True
):
    bunch = load_law_school()
    sensitive_attribute = ["race1", "gender"]
    output_variable = "bar"
    drop_columns = ["ugpagt3", "bar", *sensitive_attribute]
    df = bunch["frame"]
    df = df.dropna()
    p_attrs = df[sensitive_attribute]
    if preprocessed:
        if protected_attribute is not None:
            if protected_attribute == "race":
                ga_label = "white"
                gb_label = "non-white"
                group_a = pd.Series(get_protected_values(df, "race1", ga_label), name="group_a")
                group_b = pd.Series(get_protected_values(df, "race1", gb_label), name="group_b")

            elif protected_attribute == "gender":
                ga_label = "Female"
                gb_label = "Male"
                group_a = pd.Series(get_protected_values(df, "gender", ga_label), name="group_a")
                group_b = pd.Series(get_protected_values(df, "gender", gb_label), name="group_b")
            else:
                raise ValueError("The protected attribute must be one of: race or gender")

        y = df[output_variable]
        y = y.map({"FALSE": 0, "TRUE": 1}).astype("category")
        X = df.drop(drop_columns, axis=1)
        if protected_attribute is not None:
            metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
            return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
        return Dataset(X=X, y=y, p_attrs=p_attrs)

    y = df[output_variable]
    X = df.drop(drop_columns, axis=1)

    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_student_multiclass_dataset(
    protected_attribute: Union[Literal["sex", "address"], None] = None, preprocessed=True
):
    sensitive_attributes = ["sex", "address", "Mjob", "Fjob"]
    output_column = "G3"
    drop_columns = ["G1", "G2", "G3", "sex", "address", "Mjob", "Fjob"]
    bunch = load_student()
    df = bunch["frame"]

    # we don't want to encode protected attributes
    df = df.dropna()
    y = convert_float_to_categorical(df[output_column], 3)

    for col in ["sex", "address"]:
        df[col] = df[col].apply(lambda x: x.replace("'", ""))

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")

    p_attrs = df[sensitive_attributes]
    if preprocessed:
        if protected_attribute is not None:
            if protected_attribute == "sex":
                ga_label = "F"
                gb_label = "M"
                group_a = pd.Series(df["sex"] == ga_label, name="group_a")
                group_b = ~group_a
            elif protected_attribute == "address":
                ga_label = "U"
                gb_label = "M"
                group_a = pd.Series(df["address"] == ga_label, name="group_a")
                group_b = ~group_a
            else:
                raise ValueError("The protected attribute must be one sex or address")

        for col in df.select_dtypes(include=["category"]):
            df[col] = pd.factorize(df[col])[0]

        df = df.drop(drop_columns, axis=1)
        df = pd.get_dummies(df, columns=df.columns[df.dtypes == "category"])
        df = df.reset_index(drop=True)
        X = df.astype(float)
        y = y.reset_index(drop=True)
        if protected_attribute is not None:
            metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
            return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
        return Dataset(X=X, y=y, p_attrs=p_attrs)

    X = df.drop(columns=drop_columns)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_student_dataset(
    target: Literal["G1", "G2", "G3"] = "G3",
    preprocessed: bool = True,
    protected_attribute: Union[Literal["sex", "address"], None] = None,
):
    sensitive_attributes = ["sex", "address", "Mjob", "Fjob"]
    drop_columns = ["G1", "G2", "G3", "sex", "address", "Mjob", "Fjob"]
    bunch = load_student()

    df = bunch["frame"]
    for col in ["sex", "address", "Mjob", "Fjob"]:
        df[col] = df[col].apply(lambda x: x.replace("'", ""))

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")

    df = df.dropna()
    y = df[target]
    p_attrs = df[sensitive_attributes]
    if preprocessed:
        if protected_attribute is not None:
            if protected_attribute == "sex":
                ga_label = "'F'"
                gb_label = "'M'"
                group_a = pd.Series(df["sex"] == "'F'", name="group_a")
                group_b = pd.Series(df["sex"] == "'M'", name="group_b")
            else:
                raise ValueError("The protected attribute doesn't exist or not implemented")

        for col in df.select_dtypes(include=["category"]):
            df[col] = pd.factorize(df[col])[0]
        df = df.drop(drop_columns, axis=1)
        df = pd.get_dummies(df, columns=df.columns[df.dtypes == "category"])
        df = df.reset_index(drop=True)
        X = df.astype(float)
        y = y.reset_index(drop=True)

        if protected_attribute is not None:
            metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
            return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
        return Dataset(X=X, y=y, p_attrs=p_attrs)

    X = df.drop(columns=drop_columns)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


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
    y_cat = pd.Series(convert_float_to_categorical(df[output_column], 3)).astype("category")
    remove_columns = [protected_attribute, output_column]
    x = df.drop(columns=remove_columns)
    numeric_features = x.select_dtypes(include=[np.number]).columns
    return Dataset(X=x[numeric_features], y=y_cat, group_a=group_a, group_b=group_b, p_attr=p_attr)


def load_clinical_records_dataset(preprocessed: bool = True, protected_attribute: Union[Literal["sex"], None] = None):
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
    protected_attributes = ["age", "sex"]
    output_variable = "DEATH_EVENT"
    drop_columns = ["age", "sex", "DEATH_EVENT"]
    p_attrs = df[protected_attributes]
    df = df.dropna().reset_index(drop=True)
    if preprocessed:
        if protected_attribute is not None:
            if protected_attribute == "sex":
                ga_label = 0
                gb_label = 1
                group_a = pd.Series(df[protected_attribute] == ga_label, name="group_a")
                group_b = pd.Series(df[protected_attribute] == gb_label, name="group_b")
            else:
                raise ValueError("The protected attribute doesn't exist or not implemented")
        y = df[output_variable]
        X = df.drop(columns=drop_columns)
        if protected_attribute is not None:
            metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
            return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
        return Dataset(X=X, y=y, p_attrs=p_attrs)

    y = df[output_variable]
    x = df.drop(columns=drop_columns)
    return Dataset(X=x, y=y, p_attrs=p_attrs)


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
        return load_student_multiclass_dataset(**kargs)
    if dataset_name == "student":
        return load_student_dataset(**kargs)
    if dataset_name == "lastfm":
        return load_lastfm_dataset()
    if dataset_name == "us_crime":
        return load_us_crime_dataset()
    if dataset_name == "us_crime_multiclass":
        return load_us_crime_multiclass_dataset()
    if dataset_name == "clinical_records":
        return load_clinical_records_dataset(**kargs)
    raise NotImplementedError
