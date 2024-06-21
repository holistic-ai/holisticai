from __future__ import annotations

from typing import TYPE_CHECKING, Union

import pandas as pd
from holisticai.xai.commons._definitions import (
    LearningTaskXAISettings,
    SurrogateFeatureImportance,
)
from numpy.random import RandomState
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import accuracy_score, mean_squared_error

if TYPE_CHECKING:
    from holisticai.datasets import Dataset

metric_scores = {
    "binary_classification": accuracy_score,
    "regression": mean_squared_error,
    "multi_classification": accuracy_score,
}


class SurrogateFeatureImportanceCalculator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    learning_task_settings: LearningTaskXAISettings
    random_state: Union[RandomState, int] = RandomState(42)

    def create_surrogate_model(self, learning_task, x, y):
        if learning_task in ["binary_classification", "multi_classification"]:
            from sklearn.tree import DecisionTreeClassifier

            dt = DecisionTreeClassifier(max_depth=3, random_state=self.random_state)
            return dt.fit(x, y)

        if learning_task == "regression":
            from sklearn.tree import DecisionTreeRegressor

            dt = DecisionTreeRegressor(max_depth=3, random_state=self.random_state)
            return dt.fit(x, y)

        raise ValueError("model_type must be either 'binary_classification', 'multi_classification' or 'regression'")

    def __call__(self, ds: Dataset) -> SurrogateFeatureImportance:
        """
        Compute surrogate feature importance for a given model type, model and input features.

        Args:
            model_type (str): The type of the model, either 'binary_classification' or 'regression'.
            model (sklearn estimator): The model to compute surrogate feature importance for.
            x (pandas.DataFrame): The input features.

        Returns:
            holisticai.explainability.feature_importance.SurrogateFeatureImportance: The surrogate feature importance.
        """
        X = ds["X"]  # noqa: N806
        y_pred = self.learning_task_settings.predict_fn(X)
        surrogate = self.create_surrogate_model(self.learning_task_settings.learning_task, X, y_pred)
        feature_names = X.columns
        importances = surrogate.feature_importances_
        feature_importances = pd.DataFrame.from_dict(
            {"Variable": feature_names, "Importance": importances}
        ).sort_values("Importance", ascending=False)
        return SurrogateFeatureImportance(feature_importances=feature_importances, surrogate=surrogate)
