from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.metrics import accuracy_score, mean_squared_error

from holisticai.utils import ConditionalImportances, Importances, ModelProxy
from holisticai.utils.feature_importances import group_samples_by_learning_task

metric_scores = {
    "binary_classification": accuracy_score,
    "regression": mean_squared_error,
    "multi_classification": accuracy_score,
}


def compute_surrogate_feature_importance(
    proxy: ModelProxy,
    X: pd.DataFrame,
    y: Union[pd.Series, None] = None,
    random_state: Union[RandomState, int, None] = None,
    conditional: bool = False,
) -> Union[Importances, ConditionalImportances]:
    if conditional and y is None:
        raise ValueError("y must be provided when conditional=True")

    pfi = SurrogateFeatureImportanceCalculator(random_state=random_state)
    if conditional:
        sample_groups = group_samples_by_learning_task(y, proxy.learning_task)
        values = {
            group_name: pfi.compute_importances(X=X.loc[indexes], proxy=proxy)
            for group_name, indexes in sample_groups.items()
        }
        return ConditionalImportances(values=values)
    return pfi.compute_importances(X=X, proxy=proxy)


class SurrogateFeatureImportanceCalculator:
    def __init__(self, random_state: Union[RandomState, int, None] = None, importance_type: str = "global"):
        if random_state is None:
            random_state = RandomState(42)
        self.random_state = random_state
        self.importance_type = importance_type

    def create_surrogate_model(self, X: pd.DataFrame, y: pd.Series, learning_task: str):
        if learning_task in ["binary_classification", "multi_classification"]:
            from sklearn.tree import DecisionTreeClassifier

            dt = DecisionTreeClassifier(max_depth=3, random_state=self.random_state)
            return dt.fit(X, y)

        if learning_task == "regression":
            from sklearn.tree import DecisionTreeRegressor

            dt = DecisionTreeRegressor(max_depth=3, random_state=self.random_state)
            return dt.fit(X, y)

        raise ValueError("model_type must be either 'binary_classification', 'multi_classification' or 'regression'")

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
        return Importances(values=importances, feature_names=feature_names, extra_attrs={"surrogate": surrogate})
