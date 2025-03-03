from __future__ import annotations

from typing import Any, Literal, Union, overload

import numpy as np
import pandas as pd
from holisticai.inspection._utils import group_index_samples_by_learning_task
from holisticai.utils._commons import get_columns, get_item
from holisticai.utils._definitions import (
    ConditionalImportances,
    Importances,
    ModelProxy,
)
from numpy.random import RandomState
from sklearn.metrics import accuracy_score, mean_squared_error

metric_scores = {
    "binary_classification": accuracy_score,
    "regression": mean_squared_error,
    "multi_classification": accuracy_score,
}


@overload
def compute_permutation_importance(
    proxy: ModelProxy,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    n_jobs: int = -1,
    random_state: Union[RandomState, int, None] = None,
    importance_type: Literal["conditional"] = "conditional",
) -> ConditionalImportances: ...


@overload
def compute_permutation_importance(
    proxy: ModelProxy,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    n_jobs: int = -1,
    random_state: Union[RandomState, int, None] = None,
) -> Importances: ...


def compute_permutation_importance(
    proxy: ModelProxy,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    n_jobs: int = -1,
    random_state: Union[RandomState, int, None] = None,
    importance_type: Literal["standard", "conditional"] = "standard",
) -> Union[Importances, ConditionalImportances]:
    pfi = PermutationFeatureImportanceCalculator(n_repeats=n_repeats, n_jobs=n_jobs, random_state=random_state)
    if importance_type == "conditional":
        return compute_conditional_permutation_importance(
            proxy=proxy, X=X, y=y, n_repeats=n_repeats, n_jobs=n_jobs, random_state=random_state
        )
    return pfi.compute_importances(X=X, y=y, proxy=proxy)


def compute_conditional_permutation_importance(
    proxy: ModelProxy,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    n_jobs: int = -1,
    random_state: Union[RandomState, int, None] = None,
) -> Union[Importances, ConditionalImportances]:
    pfi = PermutationFeatureImportanceCalculator(n_repeats=n_repeats, n_jobs=n_jobs, random_state=random_state)
    sample_groups = group_index_samples_by_learning_task(y, proxy.learning_task)
    values = {}
    for group_name, indexes in sample_groups.items():
        values[group_name] = pfi.compute_importances(X=get_item(X, indexes), y=get_item(y, indexes), proxy=proxy)

    return ConditionalImportances(values=values)


class SklearnClassifier:
    @staticmethod
    def from_proxy(proxy):
        class Wrapper:
            predict = proxy.predict
            predict_proba = proxy.predict_proba
            classes = proxy.classes
            fit = lambda x, y: None  # noqa: ARG005, E731
            score = lambda x, y: accuracy_score(y, proxy.predict(x))  # noqa: E731

        return Wrapper


class SklearnRegressor:
    @staticmethod
    def from_proxy(proxy):
        class Wrapper:
            predict = proxy.predict
            fit = lambda x, y: None  # noqa: ARG005, E731
            score = lambda x, y: mean_squared_error(y, proxy.predict(x))  # noqa: E731

        return Wrapper


class PermutationFeatureImportanceCalculator:
    def __init__(
        self,
        n_repeats: int = 5,
        n_jobs: int = -1,
        random_state: Union[RandomState, int, None] = None,
        importance_type: str = "global",
    ):
        if random_state is None:
            random_state = RandomState(42)
        if isinstance(random_state, int):
            random_state = RandomState(random_state)
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.importance_type = importance_type

    def compute_importances(self, X: Any, y: Any, proxy: ModelProxy) -> Importances:
        if proxy.learning_task == "regression":
            sklearn_model = SklearnRegressor.from_proxy(proxy)

        elif proxy.learning_task in ("binary_classification", "multi_classification"):
            sklearn_model = SklearnClassifier.from_proxy(proxy)

        from sklearn.inspection import permutation_importance

        r = permutation_importance(sklearn_model, X, y, n_repeats=self.n_repeats, random_state=0, n_jobs=self.n_jobs)
        feature_importance_values = np.abs(r["importances_mean"])

        features = list(get_columns(X))
        feature_importances = pd.DataFrame.from_dict(
            {"Variable": features, "Importance": feature_importance_values}
        ).sort_values("Importance", ascending=False)
        feature_importances["Importance"] = feature_importances["Importance"] / feature_importances["Importance"].sum()
        feature_names = list(feature_importances["Variable"].values)
        importances = np.array(feature_importances["Importance"].values)
        return Importances(values=importances, feature_names=feature_names)
