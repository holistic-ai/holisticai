from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.metrics import accuracy_score, mean_squared_error

from holisticai.datasets import Dataset
from holisticai.utils import ConditionalImportance, Importances, ModelProxy
from holisticai.utils.feature_importances import group_samples_by_learning_task

metric_scores = {
    "binary_classification": accuracy_score,
    "regression": mean_squared_error,
    "multi_classification": accuracy_score,
}


def compute_surrogate_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    proxy: ModelProxy,
    random_state: Union[RandomState, int, None] = None,
    conditional: bool = False,
) -> Union[Importances, ConditionalImportance]:
    pfi = SurrogateFeatureImportanceCalculator(random_state=random_state)
    if conditional:
        sample_groups = group_samples_by_learning_task(y, proxy.learning_task)
        values = {
            group_name: pfi.compute_importances(Dataset(X=X.loc[indexes], y=y.loc[indexes]), proxy=proxy)
            for group_name, indexes in sample_groups.items()
        }
        return ConditionalImportance(values=values)
    return pfi.compute_importances(Dataset(X=X, y=y), proxy)


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

    def compute_importances(self, dataset: Dataset, proxy: ModelProxy) -> Importances:
        """
        Compute surrogate feature importance for a given model type, model and input features.

        Args:
            model_type (str): The type of the model, either 'binary_classification' or 'regression'.
            model (sklearn estimator): The model to compute surrogate feature importance for.
            x (pandas.DataFrame): The input features.

        Returns:
            holisticai.explainability.feature_importance.SurrogateFeatureImportance: The surrogate feature importance.
        """
        X = dataset["X"]
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
