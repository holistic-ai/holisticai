from __future__ import annotations

from typing import Literal, Optional, Union, overload

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

from holisticai.inspection._utils import group_index_samples_by_learning_task
from holisticai.utils._definitions import ConditionalImportances, Importances, ModelProxy

metric_scores = {
    "binary_classification": accuracy_score,
    "regression": mean_squared_error,
    "multi_classification": accuracy_score,
}


@overload
def compute_surrogate_feature_importance(
    proxy: ModelProxy,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    random_state: Optional[Union[RandomState, int]] = None,
) -> Importances: ...


@overload
def compute_surrogate_feature_importance(
    proxy: ModelProxy,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    random_state: Union[RandomState, int, None] = None,
    importance_type: Literal["conditional"] = "conditional",
) -> Importances: ...


def compute_surrogate_feature_importance(
    proxy: ModelProxy,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    random_state: Optional[Union[RandomState, int]] = None,
    importance_type: Literal["conditional", "standard"] = "standard",
) -> Union[Importances, ConditionalImportances]:
    pfi = SurrogateFeatureImportanceCalculator(random_state=random_state)
    if importance_type == "conditional":
        y = pd.Series(proxy.predict(X)) if y is None else y
        sample_groups = group_index_samples_by_learning_task(y, proxy.learning_task)
        values = {
            group_name: pfi.compute_importances(X=X.loc[indexes], proxy=proxy)
            for group_name, indexes in sample_groups.items()
        }
        return ConditionalImportances(values=values)
    return pfi.compute_importances(X=X, proxy=proxy)


class SurrogateFeatureImportanceCalculator:
    def __init__(
        self,
        random_state: Union[RandomState, int, None] = None,
        importance_type: str = "global",
    ):
        if random_state is None:
            random_state = RandomState(42)
        self.random_state = random_state
        self.importance_type = importance_type

    def create_surrogate_model(self, X: pd.DataFrame, y: pd.Series, learning_task: str):
        best_tree = None
        if learning_task in ["binary_classification", "multi_classification"]:
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, max_depth=3)
            rf.fit(X, y)
            best_score = -np.inf
            for tree in rf.estimators_:
                predicciones = tree.predict(X)
                score = metric_scores[learning_task](y, predicciones)
                if score > best_score:
                    best_score = score
                    best_tree = tree

        elif learning_task == "regression":
            rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state, max_depth=3)
            rf.fit(X, y)
            best_score = np.inf
            for tree in rf.estimators_:
                predicciones = tree.predict(X)
                score = metric_scores[learning_task](y, predicciones)
                if score < best_score:
                    best_score = score
                    best_tree = tree

        if best_tree is None:
            raise ValueError(f"Surrogate model could not be created for learning task {learning_task}")

        return best_tree

    def compute_importances(self, X: pd.DataFrame, proxy: ModelProxy) -> Importances:
        """
        Compute surrogate feature importance for a given model type, model and input features.

        Parameters
        ----------
            x (pandas.DataFrame): The input features.
            proxy ModelProxy: The model proxy.

        Returns:
            holisticai.utils.feature_importance.SurrogateFeatureImportance: The surrogate feature importance.
        """
        y_pred = proxy.predict(X)
        surrogate = self.create_surrogate_model(X, y_pred, proxy.learning_task)
        feature_names = X.columns
        importances = surrogate.feature_importances_
        feature_importances = pd.DataFrame.from_dict(
            {"Variable": feature_names, "Importance": importances}
        ).sort_values("Importance", ascending=False)
        feature_importances["Importance"] = feature_importances["Importance"] / feature_importances["Importance"].sum()

        feature_names = list(feature_importances["Variable"].values)
        importances = np.array(feature_importances["Importance"].values)
        return Importances(
            values=importances,
            feature_names=feature_names,
            extra_attrs={"surrogate": surrogate},
        )
