from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from holisticai.datasets._dataloaders import load_hai_datasets
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


def load_adult_dataset(protected_attribute: Literal["race", "sex"] | None = "sex", preprocessed: bool = True):
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

    data = load_hai_datasets(dataset_name="adult")
    df = data.copy()
    df = df.dropna().reset_index(drop=True)
    p_attrs = df[sensitive_attribute]
    if preprocessed:
        y = df[output_variable].map({"<=50K": 0, ">50K": 1}).astype("category")
        xt = df[feature_names]
        categorical_features = xt.select_dtypes(include=["category"]).columns
        xt = pd.get_dummies(xt, columns=categorical_features).astype(np.float64)
    else:
        xt = df[feature_names].astype(column_types)
        y = df[output_variable]

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
            msg = "The protected attribute must be: race or sex"
            raise ValueError(msg)

        if protected_attribute is not None:
            metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
            return Dataset(X=xt, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=xt, y=y, p_attrs=p_attrs)


def load_law_school_dataset(protected_attribute: Literal["race", "sex"] | None = "sex", preprocessed: bool = True):
    data = load_hai_datasets(dataset_name="law_school")
    sensitive_attribute = ["race1", "gender", "age"]
    output_variable = "bar"
    drop_columns = ["ugpagt3", "bar", *sensitive_attribute]
    df = data.copy()
    df = df.dropna()
    p_attrs = df[sensitive_attribute]
    y = df[output_variable]
    if preprocessed:
        y = y.map({"FALSE": 0, "TRUE": 1}).astype("category")
    X = df.drop(drop_columns, axis=1)

    if protected_attribute is not None:
        if protected_attribute == "race":
            ga_label = "white"
            gb_label = "non-white"
            group_a = pd.Series(get_protected_values(df, "race1", ga_label), name="group_a")
            group_b = pd.Series(get_protected_values(df, "race1", gb_label), name="group_b")

        elif protected_attribute == "sex":
            ga_label = "female"
            gb_label = "male"
            group_a = pd.Series(get_protected_values(df, "gender", ga_label), name="group_a")
            group_b = pd.Series(get_protected_values(df, "gender", gb_label), name="group_b")
        else:
            msg = "The protected attribute must be one of: race or gender"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_student_multiclass_dataset(protected_attribute: Literal["sex", "address"] | None = "sex", preprocessed=True):
    sensitive_attributes = ["sex", "address", "Mjob", "Fjob"]
    output_column = "G3"
    drop_columns = ["G1", "G2", "G3", "sex", "address", "Mjob", "Fjob"]
    data = load_hai_datasets(dataset_name="student")
    df = data.copy()

    # we don't want to encode protected attributes
    df = df.dropna()
    y = convert_float_to_categorical(df[output_column], 3)

    p_attrs = df[sensitive_attributes]
    X = df.drop(drop_columns, axis=1)

    if preprocessed:
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category")

        for col in X.select_dtypes(include=["category"]):
            X[col] = pd.factorize(X[col])[0]

        X = pd.get_dummies(X, columns=X.columns[X.dtypes == "category"])
        X = X.reset_index(drop=True).astype(float)
        y = y.reset_index(drop=True)

    if protected_attribute is not None:
        for col in ["sex", "address"]:
            df[col] = df[col].apply(lambda x: x.replace("'", ""))

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
            msg = "The protected attribute must be one sex or address"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_student_dataset(
    target: Literal["G1", "G2", "G3"] | None = None,
    preprocessed: bool = True,
    protected_attribute: Literal["sex", "address"] | None = "sex",
):
    if target is None:
        target = "G3"

    sensitive_attributes = ["sex", "address", "Mjob", "Fjob"]
    drop_columns = ["G1", "G2", "G3", "sex", "address", "Mjob", "Fjob"]
    data = load_hai_datasets(dataset_name="student")

    df = data.copy()
    for col in ["sex", "address", "Mjob", "Fjob"]:
        df[col] = df[col].apply(lambda x: x.replace("'", ""))

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")

    df = df.dropna()
    y = df[target]
    p_attrs = df[sensitive_attributes]

    if protected_attribute is not None:
        if protected_attribute == "sex":
            ga_label = "F"
            gb_label = "M"
            group_a = pd.Series(df["sex"] == "F", name="group_a")
            group_b = pd.Series(df["sex"] == "M", name="group_b")
        elif protected_attribute == "address":
            ga_label = "U"
            gb_label = "R"
            group_a = pd.Series(df["address"] == "U", name="group_a")
            group_b = pd.Series(df["address"] == "R", name="group_b")
        else:
            msg = "The protected attribute doesn't exist or not implemented"
            raise ValueError(msg)
    df = df.drop(drop_columns, axis=1)

    if preprocessed:
        for col in df.select_dtypes(include=["category"]):
            df[col] = pd.factorize(df[col])[0]
        df = pd.get_dummies(df, columns=df.columns[df.dtypes == "category"])
        df = df.reset_index(drop=True)
        X = df.astype(float)
        y = y.reset_index(drop=True)
    else:
        X = df

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_lastfm_dataset():
    """
    Processes the last_fm dataset and returns the data, output variable, protected group A and protected group B as numerical arrays

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
    data = load_hai_datasets(dataset_name="lastfm")
    df = data.copy()
    user_column = "user"
    item_column = "artist"
    protected_attribute = "sex"
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


def load_us_crime_dataset(preprocessed=True, protected_attribute: Literal["race"] | None = "race"):
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
    data = load_hai_datasets(dataset_name="us_crime")
    protected_attributes = ["racePctWhite"]
    mapping_name2column = {"race": "racePctWhite"}
    protected_attribute_column = mapping_name2column.get(protected_attribute)
    output_variable = "ViolentCrimesPerPop"
    df = data.copy()
    df = df.iloc[:, [i for i, n in enumerate(df.isna().sum(axis=0).T.values) if n < min_nonan_values]]
    df = df.dropna()
    p_attrs = df[protected_attributes]
    threshold = 0.5
    y = df[output_variable]
    remove_columns = [*protected_attributes, output_variable]
    X = df.drop(columns=remove_columns)

    if preprocessed:
        numeric_features = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_features]

    if protected_attribute is not None:
        if protected_attribute == "race":
            ga_label = f"racePctWhite>{threshold}"
            gb_label = f"racePctWhite<={threshold}"
            group_a = pd.Series(df[protected_attribute_column] > threshold, name="group_a")
            group_b = pd.Series(~group_a, name="group_b")
        else:
            msg = f"The protected attribute doesn't exist or not implemented. Please use: {protected_attributes}"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_us_crime_multiclass_dataset(preprocessed=True, protected_attribute: Literal["race"] | None = "race"):
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
    data = load_hai_datasets(dataset_name="us_crime")
    protected_attributes = ["racePctWhite"]
    mapping_name2column = {"race": "racePctWhite"}
    protected_attribute_column = mapping_name2column.get(protected_attribute)
    output_column = "ViolentCrimesPerPop"
    df = data.copy()
    remove_columns = [*protected_attributes, output_column]
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    X = df.drop(columns=remove_columns)
    y_cat = pd.Series(convert_float_to_categorical(df[output_column], 3)).astype("category")

    if preprocessed:
        numeric_features = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_features]
        # df = df.iloc[:, [i for i, n in enumerate(df.isna().sum(axis=0).T.values) if n < min_nonan_values]]
        # df = df.dropna()

    p_attrs = df[protected_attributes]

    if protected_attribute is not None:
        if protected_attribute == "race":
            threshold = 0.5
            ga_label = f"racePctWhite>{threshold}"
            gb_label = f"racePctWhite<={threshold}"
            group_a = pd.Series(df[protected_attribute_column] > threshold, name="group_a")
            group_b = pd.Series(~group_a, name="group_b")
        else:
            msg = f"The protected attribute doesn't exist or not implemented. Please use: {protected_attribute}"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y_cat, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y_cat, p_attrs=p_attrs)


def load_clinical_records_dataset(protected_attribute: Literal["sex"] | None = "sex"):
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
    df = load_hai_datasets(dataset_name="clinical_records")
    protected_attributes = ["age", "sex"]
    output_variable = "DEATH_EVENT"
    drop_columns = ["age", "sex", "DEATH_EVENT"]
    p_attrs = df[protected_attributes]
    df = df.dropna().reset_index(drop=True)
    X = df.drop(columns=drop_columns)
    y = df[output_variable]

    if protected_attribute is not None:
        if protected_attribute == "sex":
            ga_label = 0
            gb_label = 1
            group_a = pd.Series(df[protected_attribute] == ga_label, name="group_a")
            group_b = pd.Series(df[protected_attribute] == gb_label, name="group_b")
        else:
            msg = "The protected attribute doesn't exist or not implemented. Please use: sex"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_german_credit_dataset(preprocessed=True, protected_attribute: Literal["sex"] | None = "sex"):
    """
    Processes the german credit dataset and returns the data, output variable, protected group A and protected group B as numerical arrays or as dataframe if needed

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
    data = load_hai_datasets(dataset_name="german_credit")
    protected_attributes = ["Sex", "Age"]
    output_column = "Risk"
    # change risk to binary
    data[output_column] = data[output_column].map({"good": 0, "bad": 1})
    df = data.copy()
    remove_columns = [*protected_attributes, output_column]
    df = df.ffill()
    df.reset_index(drop=True, inplace=True)
    X = df.drop(columns=remove_columns)

    if preprocessed:
        X = pd.get_dummies(X, columns=X.select_dtypes(include=["category", "object"]).columns, dtype=float)

    p_attrs = df[protected_attributes]
    y = df[output_column]

    if protected_attribute is not None:
        if protected_attribute == "sex":
            ga_label = "male"
            gb_label = "female"
            group_a = pd.Series(df["Sex"] == "male", name="group_a")
            group_b = pd.Series(df["Sex"] == "female", name="group_b")
        else:
            msg = f"The protected attribute doesn't exist or not implemented. Please use: {protected_attribute}"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_census_kdd_dataset(preprocessed=True, protected_attribute: Literal["sex"] | None = "sex"):
    """
    Processes the Census KDD dataset and returns the data, output variable, protected group A and protected group B as numerical arrays or as dataframe if needed

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
    data = load_hai_datasets(dataset_name="census_kdd")
    protected_attributes = ["race", "sex"]
    output_column = "income_50k"
    # change risk to binary
    data[output_column] = data[output_column].map({"' - 50000.'": 0, "' 50000+.'": 1})
    df = data.copy()
    remove_columns = [*protected_attributes, output_column]
    df = df.dropna(axis=0)
    df.reset_index(drop=True, inplace=True)
    X = df.drop(columns=remove_columns)

    if preprocessed:
        X = pd.get_dummies(X, columns=X.select_dtypes(include=["category", "object"]).columns, dtype=float)

    p_attrs = df[protected_attributes]
    y = df[output_column]

    df["race"] = [1 if x == "' White'" else 0 for x in df["race"]]

    if protected_attribute is not None:
        if protected_attribute == "sex":
            ga_label = "Male"
            gb_label = "Female"
            group_a = pd.Series(df["sex"] == "' Male'", name="group_a")
            group_b = pd.Series(df["sex"] == "' Female'", name="group_b")

        elif protected_attribute == "race":
            ga_label = "White"
            gb_label = "Non-White"
            group_a = pd.Series(df["race"] == 1, name="group_a")
            group_b = pd.Series(df["race"] == 0, name="group_b")
        else:
            msg = f"The protected attribute doesn't exist or not implemented. Please use: {protected_attribute}"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_bank_marketing_dataset(preprocessed=True, protected_attribute: Literal["marital"] | None = "marital"):
    """
    Processes the Banking Marketing dataset and returns the data, output variable, protected group A and protected group B as numerical arrays or as dataframe if needed

    Parameters
    ----------
    preprocessed : bool
        Whether to return the preprocessed X and y.
    protected_attribute : str
        If this parameter is set, the dataset will be returned with the protected attribute as a binary column group_a and group_b.
        Otherwise, the dataset will be returned with the protected attribute as a column p_attrs.

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    data = load_hai_datasets(dataset_name="bank_marketing")
    protected_attributes = ["marital", "age"]
    output_column = "class"
    data[output_column] = data[output_column].map({1: 0, 2: 1})

    rename_columns = {
        "V1": "age",
        "V2": "job",
        "V3": "marital",
        "V4": "education",
        "V5": "default",
        "V6": "balance",
        "V7": "housing",
        "V8": "loan",
        "V9": "contact",
        "V10": "day",
        "V11": "month",
        "V12": "duration",
        "V13": "campaign",
        "V14": "pdays",
        "V15": "previous",
        "V16": "poutcome",
    }
    df = data.copy()
    df.rename(columns=rename_columns, inplace=True)

    df["marital"] = [1 if x == "married" else 0 for x in df["marital"]]
    remove_columns = [*protected_attributes, output_column]
    df = df.dropna(axis=0)
    df.reset_index(drop=True, inplace=True)
    X = df.drop(columns=remove_columns)

    if preprocessed:
        X = pd.get_dummies(X, columns=X.select_dtypes(include=["category", "object"]).columns, dtype=float)

    p_attrs = df[protected_attributes]
    y = df[output_column]

    if protected_attribute is not None:
        if protected_attribute == "marital":
            ga_label = "Married"
            gb_label = "Not Married"
            group_a = pd.Series(df["marital"] == 1, name="group_a")
            group_b = pd.Series(df["marital"] == 0, name="group_b")
        else:
            msg = f"The protected attribute doesn't exist or not implemented. Please use: {protected_attribute}"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_compas_two_year_recid_dataset(preprocessed=True, protected_attribute: Literal["race", "sex"] | None = "race"):
    """
    Processes the compas dataset and returns the data, output variable, protected group A and protected group B as numerical arrays or as dataframe if needed
    Target: 2-year recidivism

    Parameters
    ----------
    preprocessed : bool
        Whether to return the preprocessed X and y.
    protected_attribute : str
        If this parameter is set, the dataset will be returned with the protected attribute as a binary column group_a and group_b.
        Otherwise, the dataset will be returned with the protected attribute as a column p_attrs.

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """

    data = load_hai_datasets(dataset_name="compas_two_year_recid")
    protected_attributes = ["race", "sex", "age"]
    output_column = "two_year_recid"

    df = data.copy()
    remove_columns = [*protected_attributes, output_column]
    df.reset_index(drop=True, inplace=True)
    X = df.drop(columns=remove_columns)

    if preprocessed:
        X = pd.get_dummies(X, columns=X.select_dtypes(include=["category", "object"]).columns, dtype=float)

    p_attrs = df[protected_attributes]
    y = df[output_column]

    if protected_attribute is not None:
        if protected_attribute == "sex":
            ga_label = "Male"
            gb_label = "Female"
            group_a = pd.Series(df["sex"] == ga_label, name="group_a")
            group_b = pd.Series(df["sex"] == gb_label, name="group_b")
        elif protected_attribute == "race":
            ga_label = "Causasian"
            gb_label = "Non-Caucasian"
            group_a = pd.Series(df["race"] == 1, name="group_a")
            group_b = pd.Series(df["race"] == 2, name="group_b")
        else:
            msg = f"The protected attribute doesn't exist or not implemented. Please use: {protected_attribute}"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_compas_is_recid_dataset(preprocessed=True, protected_attribute: Literal["race", "sex"] | None = "race"):
    """
    Processes the compas dataset and returns the data, output variable, protected group A and protected group B as numerical arrays or as dataframe if needed
    Target: 2-year recidivism

    Parameters
    ----------
    preprocessed : bool
        Whether to return the preprocessed X and y.
    protected_attribute : str
        If this parameter is set, the dataset will be returned with the protected attribute as a binary column group_a and group_b.
        Otherwise, the dataset will be returned with the protected attribute as a column p_attrs.

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    data = load_hai_datasets(dataset_name="compas_is_recid")
    protected_attributes = ["race", "sex", "age"]
    output_column = "is_recid"
    df = data.copy()
    remove_columns = [*protected_attributes, output_column]
    df.reset_index(drop=True, inplace=True)
    X = df.drop(columns=remove_columns)

    if preprocessed:
        X = pd.get_dummies(X, columns=X.select_dtypes(include=["category", "object"]).columns, dtype=float)

    p_attrs = df[protected_attributes]
    y = df[output_column]

    if protected_attribute is not None:
        if protected_attribute == "sex":
            ga_label = "Male"
            gb_label = "Female"
            group_a = pd.Series(df["sex"] == ga_label, name="group_a")
            group_b = pd.Series(df["sex"] == gb_label, name="group_b")
        elif protected_attribute == "race":
            ga_label = "Causasian"
            gb_label = "Non-Caucasian"
            group_a = pd.Series(df["race"] == 1, name="group_a")
            group_b = pd.Series(df["race"] == 2, name="group_b")
        else:
            msg = f"The protected attribute doesn't exist or not implemented. Please use: {protected_attribute}"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_diabetes_dataset(preprocessed=True, protected_attribute: Literal["race", "sex"] | None = "sex"):
    """
    Processes the Diabetes dataset and returns the data, output variable, protected group A and protected group B as numerical arrays or as dataframe if needed

    Parameters
    ----------
    preprocessed : bool
        Whether to return the preprocessed X and y.
    protected_attribute : str
        If this parameter is set, the dataset will be returned with the protected attribute as a binary column group_a and group_b.
        Otherwise, the dataset will be returned with the protected attribute as a column p_attrs.

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    data = load_hai_datasets(dataset_name="diabetes")
    protected_attributes = ["race", "gender", "age"]
    output_column = "readmit_30_days"
    data[output_column] = data[output_column].astype(int)

    df = data.copy()
    df["race"] = ["Caucasian" if x == "Caucasian" else "Non-Caucasian" for x in df["race"]]
    # drop gender row with "unknown/invalid" value
    df = df[~df["gender"].isin(["Unknown/Invalid"])]
    remove_columns = [*protected_attributes, output_column]
    df.reset_index(drop=True, inplace=True)
    X = df.drop(columns=remove_columns)

    if preprocessed:
        X = pd.get_dummies(X, columns=X.select_dtypes(include=["category", "object"]).columns, dtype=float)

    p_attrs = df[protected_attributes]
    y = df[output_column]

    if protected_attribute is not None:
        if protected_attribute == "sex":
            ga_label = "Male"
            gb_label = "Female"
            group_a = pd.Series(df["gender"] == "Male", name="group_a")
            group_b = pd.Series(df["gender"] == "Female", name="group_b")
        elif protected_attribute == "race":
            ga_label = "Caucasian"
            gb_label = "Non-Caucasian"
            group_a = pd.Series(df["race"] == "Caucasian", name="group_a")
            group_b = pd.Series(df["race"] == "Non-Caucasian", name="group_b")
        else:
            msg = f"The protected attribute doesn't exist or not implemented. Please use: {protected_attribute}"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_acsincome_dataset(preprocessed=True, protected_attribute: Literal["race", "sex"] | None = "sex"):
    """
    Processes the ACSIncome dataset and returns the data, output variable, protected group A and protected group B as numerical arrays or as dataframe if needed

    Parameters
    ----------
    preprocessed : bool
        Whether to return the preprocessed X and y.
    protected_attribute : str
        If this parameter is set, the dataset will be returned with the protected attribute as a binary column group_a and group_b.
        Otherwise, the dataset will be returned with the protected attribute as a column p_attrs.

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    data = load_hai_datasets(dataset_name="acsincome")
    protected_attributes = ["AGEP", "RAC1P", "SEX"]
    output_column = "PINCP"  # Total person's income
    data[output_column] = (data[output_column] > 50000).astype(int)  # map income to binary

    df = data.copy()
    df["RAC1P"] = ["White" if x == 1 else "Non-White" for x in df["RAC1P"]]
    remove_columns = [*protected_attributes, output_column]
    df.reset_index(drop=True, inplace=True)
    X = df.drop(columns=remove_columns)

    if preprocessed:
        X = pd.get_dummies(X, columns=X.select_dtypes(include=["category", "object"]).columns, dtype=float)

    p_attrs = df[protected_attributes]
    y = df[output_column]

    if protected_attribute is not None:
        if protected_attribute == "sex":
            ga_label = "Male"
            gb_label = "Female"
            group_a = pd.Series(df["SEX"] == 1, name="group_a")
            group_b = pd.Series(df["SEX"] == 2, name="group_b")
        elif protected_attribute == "race":
            ga_label = "White"
            gb_label = "Non_White"
            group_a = pd.Series(df["RAC1P"] == "White", name="group_a")
            group_b = pd.Series(df["RAC1P"] == "Non-White", name="group_b")
        else:
            msg = f"The protected attribute doesn't exist or not implemented. Please use: {protected_attribute}"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_acspublic_dataset(preprocessed=True, protected_attribute: Literal["race", "sex"] | None = "sex"):
    """
    Processes the ACSPublicCoverage dataset and returns the data, output variable, protected group A and protected group B as numerical arrays or as dataframe if needed

    Parameters
    ----------
    preprocessed : bool
        Whether to return the preprocessed X and y.
    protected_attribute : str
        If this parameter is set, the dataset will be returned with the protected attribute as a binary column group_a and group_b.
        Otherwise, the dataset will be returned with the protected attribute as a column p_attrs.

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    data = load_hai_datasets(dataset_name="acspublic")
    protected_attributes = ["AGEP", "RAC1P", "SEX"]
    output_column = "PUBCOV"  # public health coverage : an individual's label is 1 if PUBCOV == 1 (with public health coverage), otherwise 0.

    df = data.copy()
    df["RAC1P"] = ["White" if x == 1 else "Non-White" for x in df["RAC1P"]]
    remove_columns = [*protected_attributes, output_column]
    df.reset_index(drop=True, inplace=True)
    X = df.drop(columns=remove_columns)

    if preprocessed:
        X = pd.get_dummies(X, columns=X.select_dtypes(include=["category", "object"]).columns, dtype=float)

    p_attrs = df[protected_attributes]
    y = df[output_column]

    if protected_attribute is not None:
        if protected_attribute == "sex":
            ga_label = "Male"
            gb_label = "Female"
            group_a = pd.Series(df["SEX"] == 1, name="group_a")
            group_b = pd.Series(df["SEX"] == 2, name="group_b")
        elif protected_attribute == "race":
            ga_label = "White"
            gb_label = "Non_White"
            group_a = pd.Series(df["RAC1P"] == "White", name="group_a")
            group_b = pd.Series(df["RAC1P"] == "Non-White", name="group_b")
        else:
            msg = f"The protected attribute doesn't exist or not implemented. Please use: {protected_attribute}"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_mw_medium_dataset(preprocessed=True, protected_attribute: Literal["race", "sex"] | None = "race"):
    """
    Processes the Minimum Wage Medium dataset and returns the data, output variable, protected group A and protected group B as numerical arrays or as dataframe if needed
    Target: contracted_hours

    Parameters
    ----------
    preprocessed : bool
        Whether to return the preprocessed X and y.
    protected_attribute : str
        If this parameter is set, the dataset will be returned with the protected attribute as a binary column group_a and group_b.
        Otherwise, the dataset will be returned with the protected attribute as a column p_attrs.

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    data = load_hai_datasets(dataset_name="mw_medium")
    protected_attributes = ["race", "sex", "age_group", "education_level", "age"]
    output_column = "class"

    df = data.copy()
    remove_columns = [*protected_attributes, output_column]
    df.reset_index(drop=True, inplace=True)
    X = df.drop(columns=remove_columns)

    if preprocessed:
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = pd.factorize(X[col])[0]

    p_attrs = df[protected_attributes]
    y = df[output_column]

    if protected_attribute is not None:
        if protected_attribute == "sex":
            ga_label = "Male"
            gb_label = "Female"
            group_a = pd.Series(df["sex"] == ga_label, name="group_a")
            group_b = pd.Series(df["sex"] == gb_label, name="group_b")
        elif protected_attribute == "race":
            ga_label = "White"
            gb_label = "Black"
            group_a = pd.Series(df["race"] == ga_label, name="group_a")
            group_b = pd.Series(df["race"] == gb_label, name="group_b")
        else:
            msg = f"The protected attribute doesn't exist or not implemented. Please use: {protected_attribute}"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_mw_small_dataset(preprocessed=True, protected_attribute: Literal["race", "sex"] | None = "race"):
    """
    Processes the Minimum Wage Small dataset and returns the data, output variable, protected group A and protected group B as numerical arrays or as dataframe if needed
    Target: contracted_hours

    Parameters
    ----------
    preprocessed : bool
        Whether to return the preprocessed X and y.
    protected_attribute : str
        If this parameter is set, the dataset will be returned with the protected attribute as a binary column group_a and group_b.
        Otherwise, the dataset will be returned with the protected attribute as a column p_attrs.

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    data = load_hai_datasets(dataset_name="mw_small")
    protected_attributes = ["race", "sex", "age_group", "education_level", "age"]
    output_column = "class"

    df = data.copy()
    remove_columns = [*protected_attributes, output_column]
    df.reset_index(drop=True, inplace=True)
    X = df.drop(columns=remove_columns)

    if preprocessed:
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = pd.factorize(X[col])[0]

    p_attrs = df[protected_attributes]
    y = df[output_column]

    if protected_attribute is not None:
        if protected_attribute == "sex":
            ga_label = "Male"
            gb_label = "Female"
            group_a = pd.Series(df["sex"] == ga_label, name="group_a")
            group_b = pd.Series(df["sex"] == gb_label, name="group_b")
        elif protected_attribute == "race":
            ga_label = "White"
            gb_label = "Black"
            group_a = pd.Series(df["race"] == ga_label, name="group_a")
            group_b = pd.Series(df["race"] == gb_label, name="group_b")
        else:
            msg = f"The protected attribute doesn't exist or not implemented. Please use: {protected_attribute}"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_mw_medium_dataset(preprocessed=True, protected_attribute: Optional[Literal["race", "sex"]] = "race"):
    """
    Processes the Minimum Wage Medium dataset and returns the data, output variable, protected group A and protected group B as numerical arrays or as dataframe if needed
    Target: contracted_hours

    Parameters
    ----------
    preprocessed : bool
        Whether to return the preprocessed X and y.
    protected_attribute : str
        If this parameter is set, the dataset will be returned with the protected attribute as a binary column group_a and group_b.
        Otherwise, the dataset will be returned with the protected attribute as a column p_attrs.

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    data = load_hai_datasets(dataset_name="mw_medium")
    protected_attributes = ["race", "sex", "age_group", "education_level", "age"]
    output_column = "class"

    df = data.copy()
    remove_columns = [*protected_attributes, output_column]
    df.reset_index(drop=True, inplace=True)
    X = df.drop(columns=remove_columns)

    if preprocessed:
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = pd.factorize(X[col])[0]

    p_attrs = df[protected_attributes]
    y = df[output_column]

    if protected_attribute is not None:
        if protected_attribute == "sex":
            ga_label = "Male"
            gb_label = "Female"
            group_a = pd.Series(df["sex"] == ga_label, name="group_a")
            group_b = pd.Series(df["sex"] == gb_label, name="group_b")
        elif protected_attribute == "race":
            ga_label = "White"
            gb_label = "Black"
            group_a = pd.Series(df["race"] == ga_label, name="group_a")
            group_b = pd.Series(df["race"] == gb_label, name="group_b")
        else:
            raise ValueError(
                f"The protected attribute doesn't exist or not implemented. Please use: {protected_attribute}"
            )

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_mw_small_dataset(preprocessed=True, protected_attribute: Optional[Literal["race", "sex"]] = "race"):
    """
    Processes the Minimum Wage Small dataset and returns the data, output variable, protected group A and protected group B as numerical arrays or as dataframe if needed
    Target: contracted_hours

    Parameters
    ----------
    preprocessed : bool
        Whether to return the preprocessed X and y.
    protected_attribute : str
        If this parameter is set, the dataset will be returned with the protected attribute as a binary column group_a and group_b.
        Otherwise, the dataset will be returned with the protected attribute as a column p_attrs.

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    data = load_hai_datasets(dataset_name="mw_small")
    protected_attributes = ["race", "sex", "age_group", "education_level", "age"]
    output_column = "class"

    df = data.copy()
    remove_columns = [*protected_attributes, output_column]
    df.reset_index(drop=True, inplace=True)
    X = df.drop(columns=remove_columns)

    if preprocessed:
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = pd.factorize(X[col])[0]

    p_attrs = df[protected_attributes]
    y = df[output_column]

    if protected_attribute is not None:
        if protected_attribute == "sex":
            ga_label = "Male"
            gb_label = "Female"
            group_a = pd.Series(df["sex"] == ga_label, name="group_a")
            group_b = pd.Series(df["sex"] == gb_label, name="group_b")
        elif protected_attribute == "race":
            ga_label = "White"
            gb_label = "Black"
            group_a = pd.Series(df["race"] == ga_label, name="group_a")
            group_b = pd.Series(df["race"] == gb_label, name="group_b")
        else:
            msg = f"The protected attribute doesn't exist or not implemented. Please use: {protected_attribute}"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_mw_medium_dataset(preprocessed=True, protected_attribute: Literal["race", "sex"] | None = "race"):
    """
    Processes the Minimum Wage Medium dataset and returns the data, output variable, protected group A and protected group B as numerical arrays or as dataframe if needed
    Target: contracted_hours

    Parameters
    ----------
    preprocessed : bool
        Whether to return the preprocessed X and y.
    protected_attribute : str
        If this parameter is set, the dataset will be returned with the protected attribute as a binary column group_a and group_b.
        Otherwise, the dataset will be returned with the protected attribute as a column p_attrs.

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    data = load_hai_datasets(dataset_name="mw_medium")
    protected_attributes = ["race", "sex", "age_group", "education_level", "age"]
    output_column = "class"

    df = data.copy()
    remove_columns = [*protected_attributes, output_column]
    df.reset_index(drop=True, inplace=True)
    X = df.drop(columns=remove_columns)

    if preprocessed:
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = pd.factorize(X[col])[0]

    p_attrs = df[protected_attributes]
    y = df[output_column]

    if protected_attribute is not None:
        if protected_attribute == "sex":
            ga_label = "Male"
            gb_label = "Female"
            group_a = pd.Series(df["sex"] == ga_label, name="group_a")
            group_b = pd.Series(df["sex"] == gb_label, name="group_b")
        elif protected_attribute == "race":
            ga_label = "White"
            gb_label = "Black"
            group_a = pd.Series(df["race"] == ga_label, name="group_a")
            group_b = pd.Series(df["race"] == gb_label, name="group_b")
        else:
            msg = f"The protected attribute doesn't exist or not implemented. Please use: {protected_attribute}"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


def load_mw_small_dataset(preprocessed=True, protected_attribute: Literal["race", "sex"] | None = "race"):
    """
    Processes the Minimum Wage Small dataset and returns the data, output variable, protected group A and protected group B as numerical arrays or as dataframe if needed
    Target: contracted_hours

    Parameters
    ----------
    preprocessed : bool
        Whether to return the preprocessed X and y.
    protected_attribute : str
        If this parameter is set, the dataset will be returned with the protected attribute as a binary column group_a and group_b.
        Otherwise, the dataset will be returned with the protected attribute as a column p_attrs.

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    data = load_hai_datasets(dataset_name="mw_small")
    protected_attributes = ["race", "sex", "age_group", "education_level", "age"]
    output_column = "class"

    df = data.copy()
    remove_columns = [*protected_attributes, output_column]
    df.reset_index(drop=True, inplace=True)
    X = df.drop(columns=remove_columns)

    if preprocessed:
        for col in X.select_dtypes(include=["category", "object"]).columns:
            X[col] = pd.factorize(X[col])[0]

    p_attrs = df[protected_attributes]
    y = df[output_column]

    if protected_attribute is not None:
        if protected_attribute == "sex":
            ga_label = "Male"
            gb_label = "Female"
            group_a = pd.Series(df["sex"] == ga_label, name="group_a")
            group_b = pd.Series(df["sex"] == gb_label, name="group_b")
        elif protected_attribute == "race":
            ga_label = "White"
            gb_label = "Black"
            group_a = pd.Series(df["race"] == ga_label, name="group_a")
            group_b = pd.Series(df["race"] == gb_label, name="group_b")
        else:
            msg = f"The protected attribute doesn't exist or not implemented. Please use: {protected_attribute}"
            raise ValueError(msg)

    if protected_attribute is not None:
        metadata = f"""{protected_attribute}: {{'group_a': '{ga_label}', 'group_b': '{gb_label}'}}"""
        return Dataset(X=X, y=y, p_attrs=p_attrs, group_a=group_a, group_b=group_b, _metadata=metadata)
    return Dataset(X=X, y=y, p_attrs=p_attrs)


ProcessedDatasets = Literal[
    "adult",
    "law_school",
    "student_multiclass",
    "student",
    "lastfm",
    "us_crime",
    "us_crime_multiclass",
    "clinical_records",
    "german_credit",
    "census_kdd",
    "bank_marketing",
    "compas_two_year_recid",
    "compas_is_recid",
    "diabetes",
    "acsincome",
    "acspublic",
    "mw_medium",
    "mw_small",
]


def load_dataset(
    dataset_name: ProcessedDatasets,
    preprocessed: bool = True,
    protected_attribute: str | None = None,
    target: str | None = None,
) -> Dataset:
    """
    Load a specific dataset based on the given dataset name.

    Parameters
    ----------
    dataset_name: ProcessedDatasets
        The name of the dataset to load. The list of supported datasets are here: :ref:`processed_datasets`.
    preprocessed: (bool, Optional)
        Whether to return the preprocessed X and y.
    protected_attribute: (str, Optional)
        If this parameter is set, the dataset will be returned with the protected attribute as a binary column group_a and group_b.
        Otherwise, the dataset will be returned with the protected attribute as a column p_attrs.

    Returns
    -------
    Dataset: The loaded dataset.

    Raises
    ------
    NotImplementedError:
        If the specified dataset name is not supported.
    """
    if dataset_name == "adult":
        return load_adult_dataset(preprocessed=preprocessed, protected_attribute=protected_attribute)
    if dataset_name == "law_school":
        return load_law_school_dataset(preprocessed=preprocessed, protected_attribute=protected_attribute)
    if dataset_name == "student_multiclass":
        return load_student_multiclass_dataset(preprocessed=preprocessed, protected_attribute=protected_attribute)
    if dataset_name == "student":
        return load_student_dataset(preprocessed=preprocessed, protected_attribute=protected_attribute, target=target)
    if dataset_name == "lastfm":
        return load_lastfm_dataset()
    if dataset_name == "us_crime":
        return load_us_crime_dataset(preprocessed=preprocessed, protected_attribute=protected_attribute)
    if dataset_name == "us_crime_multiclass":
        return load_us_crime_multiclass_dataset(preprocessed=preprocessed, protected_attribute=protected_attribute)
    if dataset_name == "clinical_records":
        return load_clinical_records_dataset(protected_attribute=protected_attribute)
    if dataset_name == "german_credit":
        return load_german_credit_dataset(preprocessed=preprocessed, protected_attribute=protected_attribute)
    if dataset_name == "census_kdd":
        return load_census_kdd_dataset(preprocessed=preprocessed, protected_attribute=protected_attribute)
    if dataset_name == "bank_marketing":
        return load_bank_marketing_dataset(preprocessed=preprocessed, protected_attribute=protected_attribute)
    if dataset_name == "compas_two_year_recid":
        return load_compas_two_year_recid_dataset(preprocessed=preprocessed, protected_attribute=protected_attribute)
    if dataset_name == "compas_is_recid":
        return load_compas_is_recid_dataset(preprocessed=preprocessed, protected_attribute=protected_attribute)
    if dataset_name == "diabetes":
        return load_diabetes_dataset(preprocessed=preprocessed, protected_attribute=protected_attribute)
    if dataset_name == "acsincome":
        return load_acsincome_dataset(preprocessed=preprocessed, protected_attribute=protected_attribute)
    if dataset_name == "acspublic":
        return load_acspublic_dataset(preprocessed=preprocessed, protected_attribute=protected_attribute)
    if dataset_name == "mw_medium":
        return load_mw_medium_dataset(preprocessed=preprocessed, protected_attribute=protected_attribute)
    if dataset_name == "mw_small":
        return load_mw_small_dataset(preprocessed=preprocessed, protected_attribute=protected_attribute)
    raise NotImplementedError


def _load_dataset_benchmark(
    dataset_name: ProcessedDatasets,
    preprocessed: bool = True,
) -> Dataset:
    """
    Load a specific dataset based on the given dataset name.

    Parameters
    ----------
    dataset_name: ProcessedDatasets
        The name of the dataset to load. The list of supported datasets are here: :ref:`processed_datasets`.
    preprocessed: (bool, Optional)
        Whether to return the preprocessed X and y.
    protected_attribute: (str, Optional)
        If this parameter is set, the dataset will be returned with the protected attribute as a binary column group_a and group_b.
        Otherwise, the dataset will be returned with the protected attribute as a column p_attrs.

    Returns
    -------
    Dataset: The loaded dataset.

    Raises
    ------
    NotImplementedError:
        If the specified dataset name is not supported.
    """
    if dataset_name == "adult_sex":
        return load_adult_dataset(preprocessed=preprocessed, protected_attribute="sex")
    if dataset_name == "adult_race":
        return load_adult_dataset(preprocessed=preprocessed, protected_attribute="race")

    if dataset_name == "law_school_sex":
        return load_law_school_dataset(preprocessed=preprocessed, protected_attribute="sex")
    if dataset_name == "law_school_race":
        return load_law_school_dataset(preprocessed=preprocessed, protected_attribute="race")

    if dataset_name == "student_multiclass_sex":
        return load_student_multiclass_dataset(preprocessed=preprocessed, protected_attribute="sex")
    if dataset_name == "student_multiclass_address":
        return load_student_multiclass_dataset(preprocessed=preprocessed, protected_attribute="address")

    if dataset_name == "student_sex":
        return load_student_dataset(preprocessed=preprocessed, protected_attribute="sex")
    if dataset_name == "student_address":
        return load_student_dataset(preprocessed=preprocessed, protected_attribute="address")

    if dataset_name == "us_crime_race":
        return load_us_crime_dataset(preprocessed=preprocessed, protected_attribute="race")

    if dataset_name == "us_crime_multiclass_race":
        return load_us_crime_multiclass_dataset(preprocessed=preprocessed, protected_attribute="race")

    if dataset_name == "clinical_records_sex":
        return load_clinical_records_dataset(protected_attribute="sex")

    if dataset_name == "german_credit_sex":
        return load_german_credit_dataset(preprocessed=preprocessed, protected_attribute="sex")

    if dataset_name == "census_kdd_sex":
        return load_census_kdd_dataset(preprocessed=preprocessed, protected_attribute="sex")

    if dataset_name == "bank_marketing_marital":
        return load_bank_marketing_dataset(preprocessed=preprocessed, protected_attribute="marital")

    if dataset_name == "compas_two_year_recid_sex":
        return load_compas_two_year_recid_dataset(preprocessed=preprocessed, protected_attribute="sex")

    if dataset_name == "compas_two_year_recid_race":
        return load_compas_two_year_recid_dataset(preprocessed=preprocessed, protected_attribute="race")

    if dataset_name == "compas_is_recid_sex":
        return load_compas_is_recid_dataset(preprocessed=preprocessed, protected_attribute="sex")

    if dataset_name == "compas_is_recid_race":
        return load_compas_is_recid_dataset(preprocessed=preprocessed, protected_attribute="race")

    if dataset_name == "diabetes_sex":
        return load_diabetes_dataset(preprocessed=preprocessed, protected_attribute="sex")

    if dataset_name == "diabetes_race":
        return load_diabetes_dataset(preprocessed=preprocessed, protected_attribute="race")

    if dataset_name == "acsincome_sex":
        return load_acsincome_dataset(preprocessed=preprocessed, protected_attribute="sex")

    if dataset_name == "acsincome_race":
        return load_acsincome_dataset(preprocessed=preprocessed, protected_attribute="race")

    if dataset_name == "acspublic_sex":
        return load_acspublic_dataset(preprocessed=preprocessed, protected_attribute="sex")

    if dataset_name == "acspublic_race":
        return load_acspublic_dataset(preprocessed=preprocessed, protected_attribute="race")

    if dataset_name == "mw_medium_race":
        return load_mw_medium_dataset(preprocessed=preprocessed, protected_attribute="race")
    if dataset_name == "mw_medium_sex":
        return load_mw_medium_dataset(preprocessed=preprocessed, protected_attribute="sex")

    if dataset_name == "mw_small_race":
        return load_mw_small_dataset(preprocessed=preprocessed, protected_attribute="race")
    if dataset_name == "mw_small_sex":
        return load_mw_small_dataset(preprocessed=preprocessed, protected_attribute="sex")

    raise NotImplementedError
