from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from holisticai.utils.obj_rep.object_repr import generate_html_for_generic_object

if TYPE_CHECKING:
    from collections.abc import Iterable

import json

import numpy as np
from numpy.random import RandomState


class DatasetDict(dict):
    """
    A dictionary-like class that represents a collection of datasets. Usually, the keys are train, validation, and test.

    Parameters:
    -----------
    datasets : dict
        A dictionary containing the datasets, where the keys are the names of the datasets and the values are the datasets themselves.

    Methods:
    --------

    __getitem__(key)
        Returns a Dataset object corresponding to the given key.
    """

    def __init__(self, **datasets):
        self.datasets = datasets

    def __getitem__(self, key):
        return self.datasets[key]

    def __repr__(self):
        datasets_repr = ",\n    ".join(f"{name}: {dataset}" for name, dataset in self.datasets.items())
        return f"DatasetDict({{\n    {datasets_repr}\n}})"

    def _repr_html_(self):
        nested_objs = []
        for name, dataset in self.datasets.items():
            repr_info = dataset.repr_info()
            repr_info["name"] = name
            nested_objs.append(repr_info)
        # Example usage
        obj = {"dtype": "DatasetDict", "attributes": {}, "nested_objects": nested_objs}
        return generate_html_for_generic_object(obj, feature_columns=5)


class GroupByDataset:
    """
    A class representing a Grouped Dataset.

    Parameters:
    -----------
    groupby_obj : pandas.core.groupby.GroupBy
        The pandas GroupBy object representing the grouped dataset.

    Attributes:
    -----------
    grouped_names : list
        A list of the names of the groups in the dataset.
    features : list
        A list of the unique features in the dataset.
    ngroups : int
        The number of groups in the dataset.
    random_state : numpy.random.RandomState
        The random state object used for sampling.

    """

    def __init__(self, groupby_obj):
        self.groupby_obj = groupby_obj
        self.grouped_names = [feature for feature, _ in self.groupby_obj.keys]
        self.features = self.groupby_obj.obj.columns.get_level_values("features").unique().tolist()
        self.ngroups = self.groupby_obj.ngroups
        self.random_state = np.random.RandomState()

    def head(self, k):
        """Returns the first k rows of each group in the dataset."""
        return Dataset(self.groupby_obj.head(k))

    def sample(self, n, random_state=None):
        """Returns a random sample of n rows from each group in the dataset."""
        if random_state is None:
            random_state = self.random_state
        return Dataset(
            self.groupby_obj.apply(lambda x: sample_n(x, n, random_state=random_state)).reset_index(drop=True)
        )

    def __iter__(self):
        """Iterates over the groups in the dataset."""
        for group_name, groupby_obj_batch in self.groupby_obj:
            yield group_name, Dataset(groupby_obj_batch)

    def count(self):
        dss = []
        for group_name, groupby_obj_batch in self.groupby_obj:
            data = dict(zip(self.grouped_names, group_name))
            data["group_size"] = len(groupby_obj_batch)
            dss.append(data)
        return pd.DataFrame(dss)

    def __repr__(self):
        """Returns a string representation of the GroupByDataset."""
        return (
            f"GroupByDataset({{\n"
            f"        names: {self.grouped_names},\n"
            f"        features: {self.features},\n"
            f"        count: {self.ngroups}\n"
            f"    }})"
        )

    def _repr_html_(self):
        """Returns an HTML representation of the GroupByDataset."""
        obj = {
            "dtype": "GroupByDataset",
            "attributes": {
                "count": self.ngroups,
                "grouped_names": self.grouped_names,
                "Features": [" , ".join(self.features)],
            },
        }
        return generate_html_for_generic_object(obj, feature_columns=5)


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


class DataLoader:
    """
    A class that represents a data loader for a dataset. This class is used to load the dataset in batches in a specific data type (jax, pandas, or numpy).

    Parameters
    ----------
    dataset: Dataset
        The dataset to load.
    batch_size: int
        The size of the batch.
    dtype: Literal["jax", "pandas", "numpy"]
        The data type to load the dataset in.

    Example
    -------

    >>> from holisticai.datasets import load_dataset
    >>> dataset = load_dataset("adult")
    >>> dataloader = DataLoader(dataset, batch_size=32, dtype="jax")
    >>> for batch in dataloader:
    ...     print(batch)
    """

    def __init__(self, dataset: Dataset, batch_size: int, dtype: Literal["jax", "pandas", "numpy"]):
        self.batch_size = batch_size
        self.dataset = dataset
        self.dtype = dtype
        self.num_batches = int(np.ceil(len(dataset) / batch_size))

    def batched(self):
        def batch_generator(batch_size):
            for i in range(self.num_batches):
                batch = Dataset(self.dataset.data.iloc[i * batch_size : (i + 1) * batch_size])
                yield batch

        if self.dtype == "jax":
            import jax.numpy as jnp

            def batch_generator_jax(batch_size):
                for batch in batch_generator(batch_size):
                    yield {f: jnp.array(batch[f].values) for f in batch.features}

            return batch_generator_jax(self.batch_size)

        if self.dtype == "pandas":

            def batch_generator_pandas(batch_size):
                for batch in batch_generator(batch_size):
                    yield {f: batch[f] for f in batch.features}

            return batch_generator_pandas(self.batch_size)

        if self.dtype == "numpy":

            def batch_generator_numpy(batch_size):
                for batch in batch_generator(batch_size):
                    yield {f: batch[f].values for f in batch.features}

            return batch_generator_numpy(self.batch_size)
        return batch_generator(self.batch_size)

    def __iter__(self):
        """Iterates over the batches in the dataset."""
        yield from self.batched()

    def _repr_html_(self):
        obj = {
            "dtype": "DataLoader",
            "attributes": {"Number of Batches": self.num_batches, "Batch Size": self.batch_size, "Type": self.dtype},
            "nested_objects": [
                {
                    "dtype": "Dataset",
                    "attributes": {"Number of Rows": self.dataset.num_rows, "Features": self.dataset.features},
                }
            ],
        }
        return generate_html_for_generic_object(obj, feature_columns=5)


class Dataset:
    """Represents a dataset.

    Parameters
    ----------
    data: pd.DataFrame
        The underlying data of the dataset.
    features: list[str]
        The list of features in the dataset.
    num_rows: int
        The number of rows in the dataset.
    random_state: np.random.RandomState
        The random state used for sampling.
    """

    def __update_metadata(self):
        """Updates the metadata of the dataset."""
        self.features = list(self.data.columns.get_level_values(0).unique())
        self.num_rows = len(self.data)
        self.indices = self.data.index

        features_values = self.data.columns.get_level_values("features")
        features_counts = features_values.value_counts()
        self.features_is_series = {key: (value == 1) for key, value in features_counts.items()}

    def __init__(self, _data: pd.DataFrame | None = None, _metadata=None, **kargs):
        if _data is None:
            self.data = {}
            for name, value in kargs.items():
                if isinstance(value, pd.DataFrame):
                    self.data[name] = value.reset_index(drop=True)
                elif isinstance(value, pd.Series):
                    self.data[name] = pd.Series(value.reset_index(drop=True), name=name).astype(value.dtype)
                else:
                    msg = f"Variable '{name}' is of type {type(value)}, but only pd.DataFrame and pd.Series are supported."
                    raise TypeError(msg)
            self.data = pd.concat(self.data, axis=1)
            self.data.columns = self.data.columns.set_names(["features", "subfeatures"])
            self.data.reset_index(drop=True)
        else:
            self.data = _data.reset_index(drop=True)
        self.__update_metadata()
        self._metadata = _metadata
        self.random_state = np.random.RandomState()

    def remove_columns(self, columns: str | list):
        """Returns a new dataset with the given columns removed."""
        return Dataset(self.data.drop(columns, level=0, axis=1), _metadata=self._metadata)

    def rename(self, renames):
        """Returns a new dataset with renamed columns."""
        return Dataset(self.data.rename(columns=renames, level=0), _metadata=self._metadata)

    def select(self, indices: Iterable):
        """Returns a new dataset with selected rows based on the given indices."""
        existing_indices = [idx for idx in indices if idx in self.indices]
        return Dataset(self.data.iloc[existing_indices], _metadata=self._metadata)

    def sample(self, n, random_state=None):
        """Returns a random sample of n rows from the dataset."""
        if random_state is None:
            random_state = self.random_state
        return Dataset(
            sample_n(self.data, n, random_state=random_state).reset_index(drop=True), _metadata=self._metadata
        )

    def filter(self, fn):
        """Returns a new dataset with rows filtered based on the given function."""

        def fnw(row):
            new_row = {k[0] if k[0] == k[1] else k: v for k, v in row.to_dict().items()}
            return fn(new_row)

        new_datad = self.data[self.data.apply(fnw, axis=1)]
        return Dataset(new_datad, _metadata=self._metadata)

    def groupby(self, key: list[str] | str):
        """Returns a new GroupByDataset object based on the given key."""
        if isinstance(key, list):
            key = [(key[0], key[0]), (key[1], key[1])]
        elif isinstance(key, str):
            key = [(key, key)]
        else:
            raise TypeError
        return GroupByDataset(self.data.groupby(key, observed=True))

    def map(self, fn, vectorized=True):
        """Applies a function to the dataset and returns a new dataset.

        Parameters
        ----------

        fn: function
            The function to apply to the dataset.
        vectorized: bool
            Whether to apply the function in a vectorized manner or not.
        """
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
        self.data.update(updated_data)
        new_data = pd.concat([self.data, updated_data[updated_data.columns.difference(self.data.columns)]], axis=1)
        return Dataset(new_data, _metadata=self._metadata)

    def train_test_split(self, test_size=0.3, **kargs):
        """Splits the dataset into train and test datasets."""
        train_df, test_df = train_test_split(self.data, test_size=test_size, **kargs)
        train = Dataset(train_df, _metadata=self._metadata)
        test = Dataset(test_df, _metadata=self._metadata)
        return DatasetDict(train=train, test=test)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.num_rows

    def __repr__(self):
        return json.dumps(self.repr_info())

    def repr_info(self):
        return {
            "dtype": "Dataset",
            "attributes": {"Number of Rows": self.num_rows, "Features": [" , ".join(self.features)]},
            "metadata": self._metadata,
        }

    def _repr_html_(self):
        return generate_html_for_generic_object(self.repr_info(), feature_columns=5)

    def __getitem__(self, key: str | int | list):
        """Returns a subset of the dataset based on the given key."""
        if isinstance(key, str):
            feature = self.data.xs(key, level="features", axis=1)
            if feature.shape[1] == 1:
                return feature.iloc[:, 0]
            feature.columns = list(feature.columns)
            return feature

        if isinstance(key, int):
            return dataframe_to_level_dict_with_series(self.data, key)

        if isinstance(key, list) and all(isinstance(k, str) for k in key):
            subset = self.data.loc[:, key]
            subset.columns = subset.columns.droplevel("features")
            return subset

        raise NotImplementedError

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("Key must be a string.")

        feature_exists = key in self.data.columns.levels[0]
        existing_subfeatures = self.data[key].columns if feature_exists else []
        if feature_exists:
            self.data = self.data.drop(columns=key, level="features")

        if isinstance(value, pd.DataFrame):
            new_columns = pd.MultiIndex.from_product([[key], value.columns])
            value.columns = new_columns
            self.data = self.data.join(value)

        if isinstance(value, pd.Series):
            new_column = (key, value.name or len(existing_subfeatures))
            self.data[new_column] = value

        self.data.columns = self.data.columns.set_names(["features", "subfeatures"])


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


def sample_n(group: pd.DataFrame, n: int, random_state: Union[RandomState, None] = None) -> pd.DataFrame:
    if len(group) < n:
        return group
    return group.sample(n=n, replace=False, random_state=random_state)
