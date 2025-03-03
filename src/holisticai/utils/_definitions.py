from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
from holisticai.utils._commons import get_number_of_feature_above_threshold_importance
from holisticai.utils._validation import _array_like_to_numpy
from holisticai.utils.obj_rep.object_repr import ReprObj

if TYPE_CHECKING:
    from holisticai.typing import ArrayLike


class BinaryClassificationProxy(ReprObj):
    learning_task: Literal["binary_classification"] = "binary_classification"

    def __init__(
        self,
        predict: Callable,
        predict_proba: Optional[Callable] = None,
        classes: Union[list, None] = None,
    ):
        if classes is None:
            classes = [0, 1]
        self.predict = predict
        self.predict_proba = predict_proba
        self.classes = classes

    def repr_info(self):
        return {
            "dtype": "Proxy Model",
            "attributes": {
                "Learning Task": self.learning_task,
                "Classes": self.classes,
            },
        }


class MultiClassificationProxy(ReprObj):
    learning_task: Literal["multi_classification"] = "multi_classification"

    def __init__(self, predict: Callable, predict_proba: Callable, classes: list):
        self.predict = predict
        self.predict_proba = predict_proba
        self.classes = classes

    def repr_info(self):
        return {
            "dtype": "Proxy Model",
            "attributes": {
                "Learning Task": self.learning_task,
                "Number of Classes": len(self.classes),
            },
        }


class RegressionProxy(ReprObj):
    learning_task: Literal["regression"] = "regression"

    def __init__(self, predict: Callable):
        self.predict = predict

    def repr_info(self):
        return {
            "dtype": "Proxy Model",
            "attributes": {
                "Learning Task": self.learning_task,
            },
        }


class ClusteringProxy(ReprObj):
    learning_task: Literal["clustering"] = "clustering"

    def __init__(self, X, labels):
        self.X = X
        self.labels = labels

    def repr_info(self):
        return {
            "dtype": "Proxy Model",
            "attributes": {
                "Learning Task": self.learning_task,
            },
        }


ModelProxy = Union[BinaryClassificationProxy, MultiClassificationProxy, RegressionProxy, ClusteringProxy]


def create_proxy(**kargs) -> ModelProxy:
    task = kargs.get("Learning Task")
    if task == "binary_classification":
        return BinaryClassificationProxy(**kargs)
    if task == "multi_classification":
        return MultiClassificationProxy(**kargs)
    if task == "regression":
        return RegressionProxy(**kargs)
    if task == "clustering":
        return ClusteringProxy(**kargs)
    raise ValueError("Unknown learning task type")


class Importances(ReprObj):
    _theme = "blue"

    def __init__(
        self, values: ArrayLike, feature_names: list[str], extra_attrs: Union[dict, None] = None, normalize=True
    ):
        if extra_attrs is None:
            extra_attrs = {}
        values_ = np.abs(_array_like_to_numpy(values))
        if normalize:
            self.values = values_ / values_.sum()
        else:
            self.values = values_
        self.feature_names = feature_names
        self.extra_attrs = extra_attrs
        self.num_top_features = get_number_of_feature_above_threshold_importance(self.values, alpha=0.8)

    def __getitem__(self, idx: int | str) -> float:
        if isinstance(idx, int):
            return self.values[idx]
        if isinstance(idx, str):
            return self.values[self.feature_names.index(idx)]
        raise ValueError(f"Invalid index type: {type(idx)}")

    def select(self, idx: list[int]):
        data = pd.DataFrame({"feature_names": self.feature_names, "values": self.values})
        data = data.sort_values("values", ascending=False)
        new_data = data.loc[idx]
        feature_names = new_data["feature_names"].tolist()
        values = list(new_data["values"].values)
        return Importances(values=values, feature_names=feature_names, normalize=False)

    def as_dataframe(self):
        return pd.DataFrame({"Variable": self.feature_names, "Importance": self.values})

    def __len__(self):
        return len(self.feature_names)

    def top_alpha(self, alpha=0.8) -> Importances:
        num_top_features = get_number_of_feature_above_threshold_importance(self.values, alpha)
        return self.top_n(num_top_features)

    def top_n(self, n=5) -> Importances:
        assert n > 0, "No features selected"
        return self.select(list(range(n)))

    def repr_info(self):
        return {
            "dtype": "Feature Importance",
            "attributes": {
                "Number of Features": len(self.feature_names),
                "Top 80% Features": self.num_top_features,
            },
        }


class ConditionalImportances(ReprObj):
    def __init__(self, values: dict[str, Importances]):
        self.values = values

    @property
    def feature_names(self):
        return {name: importance.feature_names for name, importance in self.values.items()}

    def __iter__(self):
        return iter(self.values.items())

    def repr_info(self):
        nested_objects = [
            {
                "dtype": "Feature Importance",
                "name": f"Label: {name}",
                "attributes": {
                    "Number of Features": len(importance.feature_names),
                    "top 80% Features": importance.num_top_features,
                },
            }
            for name, importance in self.values.items()
        ]
        return {
            "dtype": "Conditional Feature Importance",
            "attributes": {},
            "metadata": {},
            "nested_objects": nested_objects,
        }


class LocalImportances(ReprObj):
    def __init__(self, data: pd.DataFrame, cond: pd.Series | None = None, metadata: pd.DataFrame | None = None):
        if cond is None:
            cond = pd.Series(["all"] * len(data))
        cond.rename("condition", inplace=True)
        self.data = pd.concat([data, cond], axis=1)
        column_names = [("DataFrame", col) for col in data.columns] + [("Serie", cond.name)]

        if metadata is not None:
            self.data = pd.concat([self.data, metadata], axis=1)
            column_names += [("Metadata", col) for col in metadata.columns]

        self.data.columns = pd.MultiIndex.from_tuples(column_names)
        self.normalize_feature_importances()

    def normalize_feature_importances(self):
        data_values = self.data["DataFrame"].values
        data_values = np.abs(data_values)
        data_values = data_values / data_values.sum(axis=1, keepdims=True)
        self.data["DataFrame"] = pd.DataFrame(data_values, columns=self.data["DataFrame"].columns)

    @property
    def values(self):
        return self.data["DataFrame"].values

    def conditional(self):
        values = {
            str(group_name): LocalImportances(data=pd.DataFrame(group_data["DataFrame"]))
            for group_name, group_data in self.data.groupby(("Serie", "condition"))
        }
        return LocalConditionalImportances(values=values)

    def __add__(self, other):
        if not isinstance(other, LocalImportances):
            raise TypeError("Both operands must be instances of LocalImportances")

        data = self.data.droplevel(0, axis=1)
        serie = data["condition"]
        data.drop("condition", axis=1, inplace=True)

        other_data = other.data.droplevel(0, axis=1)
        other_serie = other_data["condition"]
        other_data.drop("condition", axis=1, inplace=True)

        # Concatenate the data and cond parts
        new_data = pd.concat([data, other_data], axis=0).reset_index(drop=True)
        new_serie = pd.Series(pd.concat([serie, other_serie], ignore_index=True))

        # Create a new instance of LocalImportances with the concatenated data
        return LocalImportances(data=new_data, cond=new_serie)

    @property
    def feature_names(self):
        return self.data["DataFrame"].columns.tolist()

    def to_global(self):
        fip = pd.Series(self.data["DataFrame"].mean(axis=0)).reset_index()
        fip.columns = ["feature_names", "values"]
        fip.sort_values("values", ascending=False, inplace=True)
        return Importances(
            values=pd.Series(fip["values"]).values,
            feature_names=fip["feature_names"].tolist(),
        )

    def repr_info(self):
        return {
            "dtype": "Local Feature Importance",
            "attributes": {
                "Number of Features": len(self.feature_names),
                "Number of Instances": len(self.data["DataFrame"]),
            },
        }


class LocalConditionalImportances:
    def __init__(self, values: dict[str, LocalImportances]):
        self.values = values

    def to_global(self):
        values = {group_name: lfi.to_global() for group_name, lfi in self.values.items()}
        return ConditionalImportances(values=values)

    @property
    def feature_names(self):
        return {name: importance.feature_names for name, importance in self.values.items()}

    def __len__(self):
        return len(self.values)


class PartialDependence(ReprObj):
    def __init__(self, values: list[list[dict]], feature_names: Optional[list[str]] = None):
        self.values = values
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(values))]
        self.feature_names = feature_names
        self.num_features = len(feature_names)
        self.num_labels = len(values)
        self.data_types = ["individual", "average", "grid_values"]

    def __getitem__(self, key) -> float:
        return self.values[key[0]][key[1]][key[2]][0]

    def get_value(self, feature_name: str, label: int, data_type: str):
        feature_index = self.feature_names.index(feature_name)
        return self.values[label][feature_index][data_type][0]

    def repr_info(self):
        return {
            "dtype": "Partial Dependence",
            "attributes": {
                "Number of Features": self.num_features,
                "Number of Labels": self.num_labels,
                "Elements": self.data_types,
                "key": "[label_idx, feature_idx, element]",
            },
        }
