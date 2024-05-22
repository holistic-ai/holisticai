from __future__ import annotations

from collections.abc import Iterable
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split

from holisticai.datasets._dataloaders import load_adult
from holisticai.datasets.dataset_processing_utils import (
    get_protected_values,
)


def process_adult_dataset():
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
    data = load_adult()
    protected_attribute = "sex"
    output_variable = "class"
    drop_columns = ["education", "race", "sex"]
    df = pd.concat([data["data"], data["target"]], axis=1)
    df = df.dropna().reset_index(drop=True)
    group_a = get_protected_values(df, protected_attribute, "Female")
    group_b = get_protected_values(df, protected_attribute, "Male")
    df = df.drop(drop_columns, axis=1)
    y = df.pop(output_variable).map({"<=50K": 0, ">=50K": 1})
    x = df
    return x, y, group_a, group_b


class DatasetDict(dict):
    def __init__(self, **datasets):
        self.datasets = datasets

    def __getitem__(self, key):
        return self.datasets[key]

    def __repr__(self):
        datasets_repr = ",\n    ".join(f"{name}: {dataset}" for name, dataset in self.datasets.items())
        return f"DatasetDict({{\n    {datasets_repr}\n}})"

    def _repr_html_(self):
        datasets_repr = "<br>".join(
            f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-weight: normal;'>{name}:</span> {dataset._repr_html_()}"
            for name, dataset in self.datasets.items()
        )
        return (
            f"<div style='color: gray; font-family: Arial, sans-serif; line-height: 1.5;'>"
            f"<span style='font-weight: normal;'>DatasetDict</span>{{<br>{datasets_repr}<br>}}</div>"
        )


class Dataset(dict):
    def __init__(self, **kargs):
        self.features = list(kargs.keys())
        self.data = [d.reset_index(drop=True) for d in kargs.values()]
        self.num_rows = len(self.data)

    def train_test_split(self, test_size=0.3, **kargs):
        splits = train_test_split(*self.data, test_size=test_size, **kargs)
        train = Dataset(**dict(zip(self.features, splits[::2])))
        test = Dataset(**dict(zip(self.features, splits[1::2])))
        return DatasetDict(train=train, test=test)

    def __repr__(self):
        return f"Dataset({{\n" f"        features: {self.features},\n" f"        num_rows: {self.num_rows}\n" f"    }})"

    def _repr_html_(self):
        return (
            f"<div style='color: black; font-family: Arial, sans-serif; line-height: 1.5;'>"
            f"<span style='font-weight: normal;'>Dataset</span>{{<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-weight: normal;'>features:</span> {self.features},<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;<span style='font-weight: normal;'>num_rows:</span> {self.num_rows}<br>"
            f"}}<br></div>"
        )

    def __getitem__(self, key: str | int):
        if isinstance(key, str):
            return self.data[self.features.index(key)]
        if isinstance(key, int):
            return {f: v.iloc[key] for f, v in zip(self.features, self.data)}
        raise NotImplementedError

    def select(self, indices: Iterable):
        return Dataset(**{f: v.iloc[indices] for f, v in zip(self.features, self.data)})


x, y, group_a, group_b = process_adult_dataset()

ds = Dataset(x=x, y=y, group_a=group_a, group_b=group_b)
ds = ds.train_test_split(test_size=0.3)
ds = ds["train"].select([1, 2])
