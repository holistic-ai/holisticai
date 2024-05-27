from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from holisticai.datasets._dataloaders import load_adult, load_last_fm, load_law_school, load_student
from holisticai.datasets.dataset_processing_utils import (
    get_protected_values,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def load_adult_dataset():
    data = load_adult()
    protected_attribute = "sex"
    output_name = "class"
    drop_columns = ["education", "race", "sex"]
    df = pd.concat([data["data"], data["target"]], axis=1)
    df = df.dropna().reset_index(drop=True)
    group_a = get_protected_values(df, protected_attribute, "Female")
    group_b = get_protected_values(df, protected_attribute, "Male")
    df = df.drop(drop_columns, axis=1)
    y = df.pop(output_name).map({"<=50K": 0, ">50K": 1})
    df = pd.get_dummies(df, columns=df.columns[df.dtypes == "category"])
    bool_columns = df.columns[df.dtypes == "bool"]
    df[bool_columns] = df[bool_columns].astype(int)
    x = df
    data = pd.concat([x, y], axis=1)
    return Dataset(data=data, x=x, y=y, group_a=group_a, group_b=group_b)


def load_law_school_dataset():
    bunch = load_law_school()
    protected_attribute = "race1"
    output_name = "bar"
    drop_columns = ["ugpagt3", "race1", "gender", "bar"]
    df = bunch["frame"]
    df = df.dropna()
    group_a = get_protected_values(df, protected_attribute, "white")
    group_b = get_protected_values(df, protected_attribute, "non-white")
    y = df[output_name]  # binary label vector
    y = y.map({"FALSE": 0, "TRUE": 1})
    x = df.drop(drop_columns, axis=1)
    data = pd.concat([x, y], axis=1)
    return Dataset(data=data, x=x, y=y, group_a=group_a, group_b=group_b, output_name=output_name)

def load_student_multiclass_dataset():
    output_name = "G3"
    protected_attributes = ['sex', 'address', 'Mjob', 'Fjob']
    drop_columns = ["G1", "G2", "G3", 'sex', 'address', 'Mjob', 'Fjob']
    bunch = load_student()
    df = bunch["frame"]

    # we don't want to encode protected attributes
    df = df.dropna()
    y = df[output_name].to_numpy()
    buckets = np.array([8, 11, 14])
    y_cat = pd.Series((y.reshape(-1, 1) > buckets.reshape(1, -1)).sum(axis=1))
    p_attr = df[protected_attributes]
    group_a = get_protected_values(df, 'sex', "F")
    group_b = get_protected_values(df, 'sex', "M")

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
    data = pd.concat([x, y], axis=1)
    return Dataset(data=data, x=x, y=y, p_attr=p_attr, group_a=group_a, group_b=group_b, output_name=output_name)

def load_student_dataset():
    output_name = ["G1", "G2", "G3"]
    protected_attributes = ['sex', 'address', 'Mjob', 'Fjob']
    drop_columns = ["G1", "G2", "G3", 'sex', 'address', 'Mjob', 'Fjob']
    bunch = load_student()
    df = bunch["frame"]

    # we don't want to encode protected attributes
    df = df.dropna()
    y = df[output_name]
    p_attr = df[protected_attributes]
    group_a = get_protected_values(df, 'sex', "F")
    group_b = get_protected_values(df, 'sex', "M")

    for col in df.select_dtypes(include=["object"]):
        df[col] = pd.factorize(df[col])[0]
    df = df.drop(drop_columns, axis=1)
    df = pd.get_dummies(df, columns=df.columns[df.dtypes == "object"])
    df = df.reset_index(drop=True)
    p_attr = p_attr.reset_index(drop=True)
    x = df.astype(float)
    y = y.reset_index(drop=True)
    data = pd.concat([x, y], axis=1)
    return Dataset(data=data, x=x, y=y, p_attr=p_attr, group_a=group_a, group_b=group_b, output_name=output_name)


def process_lastfm_dataset():
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
    df_pivot = df_pivot.fillna(0)
    return Dataset(data=df, data_pivot=df_pivot, p_attr=pd.Series(p_attr))


def load_dataset(dataset_name):
    match dataset_name:
        case "adult":
            return load_adult_dataset()
        case "law_school":
            return load_law_school_dataset()
        case "student_multiclass":
            return load_student_multiclass_dataset()
        case "student":
            return load_student_dataset()
        case "lastfm":
            return process_lastfm_dataset()
    raise NotImplementedError

class DatasetDict(dict):
    def __init__(self, **datasets):
        self.datasets = datasets

    def __getitem__(self, key):
        return self.datasets[key]

    def __repr__(self):
        datasets_repr = ",\n    ".join(f"{name}: {dataset}" for name, dataset in self.datasets.items())
        return f"DatasetDict({{\n    {datasets_repr}\n}})"

    def _repr_html_(self):
        datasets_repr = "".join(
            f"<div style='margin: 10px 0;'>"
            f"{name}:<span style='font-weight: bold;'> {dataset.__class__.__name__}<br></span>"
            f"{dataset.repr_info()}"
            f"</div>"
            for name, dataset in self.datasets.items()
        )
        return (
            #f"<div style='background-color: #E3F2FD; border: 1px solid #00ACC1; padding: 5px; border-radius: 2px; color: black; font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif; line-height: 1.5; letter-spacing: 0.02em; max-width: 600px; margin: 10px;'>"
            f"<div style='background-color: #E3F2FD; border: 1px solid #00ACC1; padding: 20px; border-radius: 10px; color: black; font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif; line-height: 1.5; letter-spacing: 0.02em; margin: 10px; display: inline-block;'>"
            f"<span style='font-weight: bold;'>DatasetDict</span><br>{datasets_repr}</div>"
        )




class Dataset(dict):
    def update_metadata(self):
        self.features = list(self.data.keys())
        self.num_rows = len(next(iter(self.data.values())))

    def __init__(self, output_name=None, **kargs):
        self.output_name = output_name
        self.data = kargs
        for name,value in kargs.items():
            if type(value) in [pd.DataFrame, pd.Series]:
                self.data[name] = value.reset_index(drop=True)
            #elif type(value) is np.ndarray:
            #    self.data[name]=value
            else:
                print(type(value))
                raise NotImplementedError
        self.update_metadata()

    def map(self, fn):
        updated_data = fn(self.data)
        for name,value in updated_data.items():
            self.data[name] = value
        self.update_metadata()
        return self

    def train_test_split(self, test_size=0.3, **kargs):

        keys = list(self.data.keys())
        values = list(self.data.values())
        splits = train_test_split(*values, test_size=test_size, **kargs)
        train = Dataset(**dict(zip(keys, splits[::2])))
        test = Dataset(**dict(zip(keys, splits[1::2])))
        return DatasetDict(train=train, test=test)

    def __repr__(self):
        return f"Dataset({{\n" f"        features: {self.features},\n" f"        num_rows: {self.num_rows}\n" f"    }})"

    def __output_name__(self):
        return self.output_name

    def repr_info(self):
        return (
            f"<ul>"
            f"<li><span style='font-weight: normal;'>features: {self.features}</span></li>"
            f"<li><span style='font-weight: normal;'>num_rows: {self.num_rows}</span></li>"
            f"<li><span style='font-weight: normal;'>output_name: {self.output_name}</span></li>"
            f"</ul>"
        )

    def _repr_html_(self):
        return (
        #f"<div style='background-color: #E3F2FD; border: 1px solid #00ACC1; padding: 5px; border-radius: 2px; color: black; font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif; line-height: 1.5; letter-spacing: 0.02em; max-width: 600px; margin: 10px;'>"
        f"<div style='background-color: #E3F2FD; border: 1px solid #00ACC1; padding: 20px; border-radius: 10px; color: black; font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif; line-height: 1.5; letter-spacing: 0.02em; margin: 10px; display: inline-block;'>"
        f"<span style='font-weight: bold;'>Dataset</span><br>"
        f"{self.repr_info()}"
        f"</div>")

    def __getitem__(self, key: str | int):
        if isinstance(key, str):
            return self.data[key]
        if isinstance(key, int):
            return {f: v.iloc[key]  for f, v in self.data.items()}
        raise NotImplementedError

    def select(self, indices: Iterable):
        return Dataset(**{f: v.iloc[indices] for f, v in zip(self.features, self.data)})
