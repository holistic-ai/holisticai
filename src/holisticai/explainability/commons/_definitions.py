from __future__ import annotations

from typing import Annotated, Literal, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field


class BinaryClassificationXAISettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    learning_task: Literal["binary_classification"] = "binary_classification"
    predict_fn: callable
    predict_proba_fn: callable
    classes: list


class MultiClassificationXAISettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    learning_task: Literal["multi_classification"] = "multi_classification"
    predict_fn: callable
    predict_proba_fn: callable
    classes: list


class RegressionClassificationXAISettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    learning_task: Literal["regression"] = "regression"
    predict_fn: callable


LearningTaskXAISettings = Annotated[
    Union[BinaryClassificationXAISettings, MultiClassificationXAISettings, RegressionClassificationXAISettings],
    Field(discriminator="learning_task"),
]


def is_list_or_array_of_ints(obj):
    if isinstance(obj, list):
        return all(isinstance(i, int) for i in obj)
    return False


class Importances(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    values: ArrayLike
    feature_names: ArrayLike
    extra_attrs: dict = {}

    def __getitem__(self, idx: int | str | list[int]):
        if isinstance(idx, int):
            return self.values[idx]
        if isinstance(idx, str):
            return self.values[self.feature_names.index(idx)]
        if isinstance(idx, (np.ndarray, list)):
            data = pd.DataFrame({"feature_names": self.feature_names, "values": self.values})
            new_data = data.loc[idx]
            feature_names = new_data["feature_names"].tolist()
            values = new_data["values"].values
            return Importances(values=values, feature_names=feature_names)
        raise ValueError(f"Invalid index type: {type(idx)}")

    def as_dataframe(self):
        return pd.DataFrame({"Variable": self.feature_names, "Importance": self.values})

    def __len__(self):
        return len(self.feature_names)

    def top_alpha(self, alpha=0.8):
        feature_weight = self.values / self.values.sum()
        accum_feature_weight = feature_weight.cumsum()
        threshold = max(accum_feature_weight.min(), alpha)
        return self[accum_feature_weight <= threshold]


class ConditionalFeatureImportance(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    values: dict[str, Importances]

    @property
    def feature_names(self):
        return {name: importance.feature_importance for name, importance in self.values.items()}

    def __iter__(self):
        return iter(self.values.items())


class LocalImportances:
    def __init__(self, data: pd.DataFrame, cond: pd.Series | None = None):
        if cond is None:
            cond = pd.Series(["all"] * len(data))
        cond.rename("condition", inplace=True)
        self.data = pd.concat([data, cond], axis=1)
        self.data.columns = pd.MultiIndex.from_tuples(
            [("DataFrame", col) for col in data.columns] + [("Serie", cond.name)]
        )

    @property
    def values(self):
        return self.data["DataFrame"].values

    def conditional(self):
        values = {
            group_name: LocalImportances(data=group_data["DataFrame"])
            for group_name, group_data in self.data.groupby(("Serie", "condition"))
        }
        return LocalConditionalFeatureImportance(values=values)

    @property
    def feature_names(self):
        return self.data.columns.tolist()

    def to_global(self):
        fip = self.data["DataFrame"].mean(axis=0).reset_index()
        fip.columns = ["feature_names", "values"]
        fip.sort_values("values", ascending=False, inplace=True)
        return Importances(values=fip["values"].values, feature_names=fip["feature_names"].tolist())


class LocalConditionalFeatureImportance:
    def __init__(self, values: dict[str, LocalImportances]):
        self.values = values

    def to_global(self):
        values = {group_name: lfi.to_global() for group_name, lfi in self.values.items()}
        return ConditionalFeatureImportance(values=values)

    @property
    def feature_names(self):
        return {name: importance.feature_names for name, importance in self.values.items()}

    def __len__(self):
        return len(self.values)


class PartialDependence(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    values: list[dict]
