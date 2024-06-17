from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from sklearn.model_selection import train_test_split

from holisticai.utils.obj_rep.datasets import generate_html

if TYPE_CHECKING:
    from collections.abc import Iterable

import numpy as np


class DatasetDict(dict):
    def __init__(self, **datasets):
        self.datasets = datasets

    def __getitem__(self, key):
        return self.datasets[key]

    def __repr__(self):
        datasets_repr = ",\n    ".join(f"{name}: {dataset}" for name, dataset in self.datasets.items())
        return f"DatasetDict({{\n    {datasets_repr}\n}})"

    def _repr_html_(self):
        dataset_info = []
        for name, dataset in self.datasets.items():
            dataset_info.append(
                {"type": "Dataset", "name": name, "features": dataset.features, "num_rows": dataset.num_rows}
            )

        datasetdict_info = {"DatasetDict": dataset_info}
        return generate_html(datasetdict_info)


def concatenate_datasets(part_datasets: list[Dataset]):
    features = part_datasets[0].features
    return Dataset(
        **{feat: pd.concat([p[feat] for p in part_datasets], axis=0).reset_index(drop=True) for feat in features}
    )


def convert_to_pandas(data):
    if all(isinstance(i, (list, tuple)) for i in data):
        return pd.DataFrame(data)
    return pd.Series(data)


def split_dataframe_by_level(df, level=0):
    dataframes = {}
    for key in df.columns.levels[level]:
        dataframes[key] = df.xs(key, axis=1, level=level)
    return dataframes


def apply_fn_to_multilevel_df(df, fn):
    result_df = pd.DataFrame()
    for level in df.columns.levels[0]:
        subset = df.xs(level, axis=1, level=0)
        result = subset.apply(fn, axis=1, result_type="expand")
        result.columns = pd.MultiIndex.from_product([[level], result.columns])
        result_df = pd.concat([result_df, result], axis=1)
    return result_df


def sample_n(group, n, random_state=None):
    if len(group) < n:
        return group
    return group.sample(n=n, replace=False, random_state=random_state)


class GroupByDataset:
    def __init__(self, groupby_obj):
        self.groupby_obj = groupby_obj
        self.grouped_names = [feature for feature, _ in self.groupby_obj.keys]
        self.features = self.groupby_obj.obj.columns.get_level_values("features").unique().tolist()
        self.ngroups = self.groupby_obj.ngroups
        self.random_state = np.random.RandomState()

    def head(self, k):
        return Dataset(self.groupby_obj.head(k))

    def sample(self, n, random_state=None):
        if random_state is None:
            random_state = self.random_state
        return Dataset(
            self.groupby_obj.apply(lambda x: sample_n(x, n, random_state=random_state)).reset_index(drop=True)
        )

    def __iter__(self):
        for group_name, groupby_obj_batch in self.groupby_obj:
            yield group_name, Dataset(groupby_obj_batch)

    def repr_info(self):
        return {
            "GroupByDataset": {"grouped_names": self.grouped_names, "features": self.features, "ngroups": self.ngroups}
        }

    def __repr__(self):
        return (
            f"GroupByDataset({{\n"
            f"        names: {self.grouped_names},\n"
            f"        features: {self.features},\n"
            f"        count: {self.ngroups}\n"
            f"    }})"
        )

    def _repr_html_(self):
        dataset_info = self.repr_info()
        return generate_html(dataset_info)


def dataframe_to_level_dict_with_series(df, row_index):
    """
    Converts a DataFrame with two-level columns into a dictionary where:
    - Level 0 names become dictionary keys.
    - Corresponding DataFrames (without level 0) are values.
    - Level 1 names become new column names.

    Args:
        df (pandas.DataFrame): The DataFrame to convert.
        row_index (int): The index of the row to use as the Series.

    Returns:
        dict: The resulting dictionary with level 0 keys and DataFrames as values.
    """
    num_levels = 2
    if not isinstance(df.columns, pd.MultiIndex) or len(df.columns.levels) != num_levels:
        msg = "DataFrame must have MultiIndex columns with two levels."
        raise ValueError(msg)

    data = {}
    for level_0_name in df.columns.levels[0]:
        feature = df[level_0_name]
        if feature.shape[1] == 1:
            data[level_0_name] = feature.iloc[row_index, 0]
        else:
            data[level_0_name] = feature.iloc[row_index]
    return data


class Dataset:
    def update_metadata(self):
        self.features = list(self.data.columns.get_level_values(0).unique())
        self.num_rows = len(self.data)
        self.indices = self.data.index

        features_values = self.data.columns.get_level_values("features")
        features_counts = features_values.value_counts()
        self.features_is_series = {key: (value == 1) for key, value in features_counts.items()}

    def __init__(self, data: pd.DataFrame | None = None, **kargs):
        if data is None:
            self.data = {}
            for name, value in kargs.items():
                if isinstance(value, pd.DataFrame):
                    self.data[name] = value.reset_index(drop=True)
                elif isinstance(value, pd.Series):
                    self.data[name] = pd.Series(value.reset_index(drop=True), name=name)
                else:
                    msg = f"Variable '{name}' is of type {type(value)}, but only pd.DataFrame and pd.Series are supported."
                    raise TypeError(msg)
            self.data = pd.concat(self.data.values(), axis=1, keys=self.data.keys())
            self.data.columns = self.data.columns.set_names(["features", "subfeatures"])
            self.data.reset_index(drop=True)
        else:
            self.data = data.reset_index(drop=True)
        self.update_metadata()
        self.random_state = np.random.RandomState()

    def rename(self, renames):
        return Dataset(self.data.rename(columns=renames, level=0))

    def select(self, indices: Iterable):
        existing_indices = [idx for idx in indices if idx in self.indices]
        return Dataset(self.data.iloc[existing_indices])

    def sample(self, n, random_state=None):
        if random_state is None:
            random_state = self.random_state
        return Dataset(sample_n(self.data, n, random_state=random_state).reset_index(drop=True))

    def filter(self, fn):
        def fnw(row):
            new_row = {k[0] if k[0] == k[1] else k: v for k, v in row.to_dict().items()}
            return fn(new_row)

        new_datad = self.data[self.data.apply(fnw, axis=1)]
        return Dataset(new_datad)

    def groupby(self, key: list[str] | str):
        if isinstance(key, list):
            key = [(key[0], key[0]), (key[1], key[1])]
        elif isinstance(key, str):
            key = [(key, key)]
        else:
            raise TypeError
        return GroupByDataset(self.data.groupby(key, observed=True))

    def map(self, fn, vectorized=True):
        if vectorized:

            def fnw(x):
                ds = {level: x.xs(level, axis=1, level="features") for level in x.columns.levels[0]}
                return fn(ds)

            new_data = fnw(self.data)
            updated_data = pd.concat(new_data, axis=1)
            updated_data.columns = pd.MultiIndex.from_tuples(
                [(key, key) for key, serie in new_data.items()], names=["features", "subfeatures"]
            )
        else:

            def fnw(row):
                result = {}
                for upper in row.index.levels[0]:
                    sub_row = row[upper]
                    if isinstance(sub_row, pd.Series):
                        result[upper] = sub_row.to_dict() if len(sub_row) > 1 else sub_row.item()
                    elif isinstance(sub_row, pd.DataFrame):
                        sub_row = sub_row.squeeze()
                        result[upper] = sub_row.to_dict() if len(sub_row) > 1 else sub_row.squeeze().item()
                return fn(result)

            updated_data = self.data.apply(fnw, axis=1, result_type="expand")
            updated_data = pd.DataFrame(updated_data)
            new_columns = pd.MultiIndex.from_tuples(
                [(col, col) for col in updated_data.columns], names=["features", "subfeatures"]
            )
            updated_data.columns = new_columns

        new_data = self.data.combine_first(updated_data)
        return Dataset(new_data)

    def train_test_split(self, test_size=0.3, **kargs):
        train_df, test_df = train_test_split(self.data, test_size=test_size, **kargs)
        train = Dataset(train_df)
        test = Dataset(test_df)
        return DatasetDict(train=train, test=test)

    def __len__(self):
        return self.num_rows

    def __repr__(self):
        return f"Dataset({{\n" f"        features: {self.features},\n" f"        num_rows: {self.num_rows}\n" f"    }})"

    def repr_info(self):
        return {"Dataset": {"features": self.features, "num_rows": self.num_rows}}

    def _repr_html_(self):
        dataset_info = self.repr_info()
        return generate_html(dataset_info)

    def __getitem__(self, key: str | int):
        if isinstance(key, str):
            feature = self.data.xs(key, level="features", axis=1)
            if feature.shape[1] == 1:
                return feature.iloc[:, 0]
            return feature
        if isinstance(key, int):
            return dataframe_to_level_dict_with_series(self.data, key)
        raise NotImplementedError
