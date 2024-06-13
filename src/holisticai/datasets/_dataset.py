from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from holisticai.utils.obj_rep.datasets import generate_html


class DatasetDict(dict):
    def __init__(self, **datasets):
        self.datasets = datasets

    def __getitem__(self, key):
        return self.datasets[key]

    def __repr__(self):
        datasets_repr = ",\n    ".join(f"{name}: {dataset}" for name, dataset in self.datasets.items())
        return f"DatasetDict({{\n    {datasets_repr}\n}})"

    def _repr_html_(self):
        dataset_info=[]
        for name, dataset in self.datasets.items():
            dataset_info.append({'type': 'Dataset',
                                 'name': name,
                                 'features': dataset.features,
                                 'num_rows': dataset.num_rows})

        datasetdict_info = {
        'DatasetDict': dataset_info
        }
        return generate_html(datasetdict_info)

def concatenate_datasets(part_datasets : list[Dataset]):
    features = part_datasets[0].features
    return Dataset(**{feat: pd.concat([p[feat] for p in part_datasets], axis=0).reset_index(drop=True) for feat in features})

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
        result = subset.apply(fn, axis=1, result_type='expand')
        result.columns = pd.MultiIndex.from_product([[level], result.columns])
        result_df = pd.concat([result_df, result], axis=1)
    return result_df


class GroupByDataset:
    def __init__(self, groupby_obj):
        self.groupby_obj = groupby_obj

    def head(self, k):
        return Dataset(self.groupby_obj.head(k))


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

    if not isinstance(df.columns, pd.MultiIndex) or len(df.columns.levels) != 2:
        msg = "DataFrame must have MultiIndex columns with two levels."
        raise ValueError(msg)

    data = {}
    for level_0_name in df.columns.levels[0]:
        feature = df[level_0_name]
        if feature.shape[1]==1:
            data[level_0_name] = feature.iloc[row_index, 0]
        else:
            data[level_0_name] = feature.iloc[row_index]
    return data



class Dataset(dict):
    def update_metadata(self):
        self.features = list(self.data.columns.get_level_values(0).unique())
        self.num_rows = len(self.data)
        self.indices = self.data.index

    def __init__(self, data : pd.DataFrame | None = None, **kargs):
        if data is None:
            self.data = {}
            for name, value in kargs.items():
                if isinstance(value, pd.DataFrame):
                    self.data[name] = value.reset_index(drop=True)
                elif isinstance(value, pd.Series):
                    self.data[name] = pd.Series(value.reset_index(drop=True), name=name)
                else:
                    msg = f"Variable '{name}' is of type {type(value)}, but only pd.DataFrame and pd.Series are supported."  # noqa: E501
                    raise TypeError(msg)
            self.data = pd.concat(self.data.values(), axis=1, keys=self.data.keys())
            self.data.reset_index(drop=True)
        else:
            self.data = data.reset_index(drop=True)
        self.update_metadata()

    def rename(self, renames):
        return Dataset(self.data.rename(columns=renames, level=0))

    def select(self, indices: Iterable):
        existing_indices = [idx for idx in indices if idx in self.indices]
        return Dataset(self.data.iloc[existing_indices])

    def filter(self, fn):
        def fnw(row):
            new_row = {k[0] if k[0]==k[1] else k:v for k,v in row.to_dict().items()}
            return fn(new_row)

        new_datad = self.data[self.data.apply(fnw, axis=1)]
        return Dataset(new_datad)

    def groupby(self, key : list[str]|str):
        if isinstance(key,list):
            key = [(key[0],key[0]),(key[1],key[1])]
        elif isinstance(key,str):
            key = [tuple(key , key)]
        else:
            raise TypeError
        return GroupByDataset(self.data.groupby(key))

    def map(self, fn, vectorized=True): # noqa: FBT002
        def fnw(x):
            return {(k, k) if type(k) is str else k: v for k, v in fn(x).items()} # noqa: E721

        if vectorized:
            updated_data = pd.DataFrame(fnw(self.data))
            updated_data.columns = pd.MultiIndex.from_tuples(updated_data.columns)
        else:
            updated_data = self.data.apply(fnw, axis=1, result_type='expand')
            updated_data = pd.DataFrame(updated_data)

        new_data = self.data.combine_first(updated_data)
        return Dataset(new_data)

    def train_test_split(self, test_size=0.3, **kargs):
        train_df,test_df = train_test_split(self.data, test_size=test_size, **kargs)
        train = Dataset(train_df)
        test = Dataset(test_df)
        return DatasetDict(train=train, test=test)

    def __repr__(self):
        return f"Dataset({{\n" f"        features: {self.features},\n" f"        num_rows: {self.num_rows}\n" f"    }})"

    def repr_info(self):
        return {
        'Dataset': {
            'features': self.features,
            'num_rows': self.num_rows
        }
        }

    def _repr_html_(self):
        dataset_info = self.repr_info()
        return generate_html(dataset_info)

    def _repr_html_(self):
        return (
        #f"<div style='background-color: #E3F2FD; border: 1px solid #00ACC1; padding: 5px; border-radius: 2px; color: black; font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif; line-height: 1.5; letter-spacing: 0.02em; max-width: 600px; margin: 10px;'>"
        f"<div style='background-color: #E3F2FD; border: 1px solid #00ACC1; padding: 20px; border-radius: 10px; color: black; font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif; line-height: 1.5; letter-spacing: 0.02em; margin: 10px; display: inline-block;'>"  # noqa: E501
        f"<span style='font-weight: bold;'>Dataset</span><br>"
        f"{self.repr_info()}"
        f"</div>")

    def __getitem__(self, key: str | int):
        if isinstance(key, str):
            return self.data.xs(key, level=0, axis=1)
        if isinstance(key, int):
            return dataframe_to_level_dict_with_series(self.data, key)
        raise NotImplementedError
