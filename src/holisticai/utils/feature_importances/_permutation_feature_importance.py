from __future__ import annotations

import os
from typing import Literal, Union, overload, Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.random import RandomState
from sklearn.metrics import accuracy_score, mean_squared_error

from holisticai.datasets import Dataset
from holisticai.utils._definitions import (
    ConditionalImportances,
    Importances,
    ModelProxy,
)
from holisticai.utils.feature_importances import group_index_samples_by_learning_task
from holisticai.utils._commons import get_columns, get_item

metric_scores = {
    "binary_classification": accuracy_score,
    "regression": mean_squared_error,
    "multi_classification": accuracy_score,
}


@overload
def compute_permutation_feature_importance(
    proxy: ModelProxy,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    n_jobs: int = -1,
    random_state: Union[RandomState, int, None] = None,
    importance_type: Literal["conditional"] = "conditional",
) -> ConditionalImportances: ...


@overload
def compute_permutation_feature_importance(
    proxy: ModelProxy,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    n_jobs: int = -1,
    random_state: Union[RandomState, int, None] = None,
    importance_type: Literal["standard"] = "standard",
) -> Importances: ...

def compute_permutation_feature_importance(
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
        sample_groups = group_index_samples_by_learning_task(y, proxy.learning_task)
        values = {}
        for group_name, indexes in sample_groups.items():
            values[group_name] =pfi.compute_importances(X=get_item(X, indexes), y=get_item(y, indexes), proxy=proxy)

        return ConditionalImportances(values=values)
    return pfi.compute_importances(X=X, y=y, proxy=proxy)


from sklearn.metrics import accuracy_score, mean_squared_error

class SklearnClassifier:
    @staticmethod
    def from_proxy(proxy):
        class Wrapper:
            predict = proxy.predict
            predict_proba = proxy.predict_proba
            classes = proxy.classes
            fit = lambda x, y: None
            score = lambda x, y: accuracy_score(y, proxy.predict(x))
        return Wrapper

class SklearnRegressor:
    @staticmethod
    def from_proxy(proxy):
        class Wrapper:
            predict = proxy.predict
            fit = lambda x, y: None
            score = lambda x, y: mean_squared_error(y, proxy.predict(x))
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

    def compute_importances(self, X: Any, y:Any, proxy: ModelProxy) -> Importances:
        #X = dataset["X"]
        #y = dataset["y"]
        #metric = metric_scores[proxy.learning_task]
        #baseline_score = metric(y, proxy.predict(X))
        #n_features = X.shape[1]
        if proxy.learning_task == "regression":
            sklearn_model = SklearnRegressor.from_proxy(proxy)
        
        elif proxy.learning_task == "binary_classification" or proxy.learning_task == "multi_classification":
            sklearn_model = SklearnClassifier.from_proxy(proxy)
        
        from sklearn.inspection import permutation_importance

        r = permutation_importance(sklearn_model, X, y, n_repeats=self.n_repeats, random_state=0, n_jobs=self.n_jobs)
        feature_importance_values = np.abs(r['importances_mean'])

        features = list(get_columns(X))
        feature_importances = pd.DataFrame.from_dict(
            {"Variable": features, "Importance": feature_importance_values}
        ).sort_values("Importance", ascending=False)
        feature_importances["Importance"] = feature_importances["Importance"] / feature_importances["Importance"].sum()
        feature_names = list(feature_importances["Variable"].values)
        importances = np.array(feature_importances["Importance"].values)
        return Importances(values=importances, feature_names=feature_names)

    def compute_importances_(self, dataset: Dataset, proxy: ModelProxy) -> Importances:
        X = dataset["X"]
        y = dataset["y"]
        metric = metric_scores[proxy.learning_task]
        baseline_score = metric(y, proxy.predict(X))
        n_features = X.shape[1]

        def _calculate_permutation_scores(predictor, X, y, col_idx, n_repeats):
            scores = []
            X_permuted = X.copy()
            shuffling_idx = np.arange(X_permuted.shape[0])
            for _ in range(n_repeats):
                self.random_state.shuffle(shuffling_idx)
                col = X_permuted.iloc[shuffling_idx, col_idx]
                col.index = X_permuted.index
                X_permuted[X_permuted.columns[col_idx]] = col
                scores.append(metric(y, predictor(X_permuted)))
            return np.array(scores)

        n_jobs = os.cpu_count() if self.n_jobs == -1 else self.n_jobs

        scores = list(
            Parallel(n_jobs=n_jobs)(
                delayed(_calculate_permutation_scores)(
                    proxy.predict,
                    X,
                    y,
                    col_idx,
                    self.n_repeats,
                )
                for col_idx in range(n_features)
            )
        )
        feature_importances = [np.mean(np.abs(baseline_score - scores[col])) for col in range(n_features)]
        features = list(X.columns)
        feature_importances = pd.DataFrame.from_dict(
            {"Variable": features, "Importance": feature_importances}
        ).sort_values("Importance", ascending=False)
        feature_importances["Importance"] = feature_importances["Importance"] / feature_importances["Importance"].sum()
        feature_names = list(feature_importances["Variable"].values)
        importances = np.array(feature_importances["Importance"].values)
        return Importances(values=importances, feature_names=feature_names)
