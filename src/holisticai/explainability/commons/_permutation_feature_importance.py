from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from holisticai.explainability.commons._definitions import Importances, LearningTaskXAISettings
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


class PermutationFeatureImportanceCalculator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    learning_task_settings: LearningTaskXAISettings
    n_repeats: int = 10
    random_state: Union[RandomState, int] = RandomState(42)
    importance_type: str = "global"

    def __call__(self, ds: Dataset) -> Importances:
        # Ensure the random state is consistent
        rng = np.random.RandomState(self.random_state) if isinstance(self.random_state, int) else self.random_state

        X = ds["X"]
        y = ds["y"]
        metric = metric_scores[self.learning_task_settings.learning_task]
        baseline_score = metric(y, self.learning_task_settings.predict_fn(X))
        feature_importances = []

        for col in range(X.shape[1]):
            scores = np.zeros(self.n_repeats)
            for i in range(self.n_repeats):
                X_permuted = X.copy()
                X_permuted.iloc[:, col] = rng.permutation(X_permuted.iloc[:, col])
                permuted_score = metric(y, self.learning_task_settings.predict_fn(X_permuted))
                scores[i] = np.abs(baseline_score - permuted_score)
            feature_importances.append(np.mean(scores))

        features = list(X.columns)
        feature_importances = pd.DataFrame.from_dict({"Variable": features, "Importance": feature_importances}).sort_values("Importance", ascending=False)
        feature_importances["Importance"] = feature_importances["Importance"] / feature_importances["Importance"].sum()
        feature_names = list(feature_importances['Variable'].values)
        importances = np.array(feature_importances['Importance'].values)
        return Importances(values=importances, feature_names=feature_names)
