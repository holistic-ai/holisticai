from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from holisticai.xai.commons._definitions import (
    LearningTaskXAISettings,
    PermutationFeatureImportance,
)
from numpy.random import RandomState
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import accuracy_score, mean_squared_error

if TYPE_CHECKING:
    from holisticai.datasets import Dataset

metric_scores = {
        "binary_classification": accuracy_score,
        "regression": mean_squared_error,
        "multi_classification": accuracy_score
}

class PermutationFeatureImportanceCalculator(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    learning_task_settings: LearningTaskXAISettings
    n_repeats: int=10
    random_state: RandomState|int = RandomState(42)

    def __call__(self, ds: Dataset) -> PermutationFeatureImportance:
        X = ds['X']  # noqa: N806
        y = ds['y']
        metric = metric_scores[self.learning_task_settings.learning_task]
        baseline_score = metric(y, self.learning_task_settings.predict_fn(X))
        feature_importances = []
        for col in range(X.shape[1]):
            scores = np.zeros(self.n_repeats)
            for i in range(self.n_repeats):
                X_permuted = X.copy()  # noqa: N806
                X_permuted.iloc[:, col] = self.random_state.permutation(X_permuted.iloc[:, col])
                permuted_score = metric(y, self.learning_task_settings.predict_fn(X_permuted))
                scores[i] = baseline_score - permuted_score
            feature_importances.append(np.mean(scores))

        features = list(X.columns)
        feature_importance = pd.DataFrame({'Variable': features, 'Importance': feature_importances})
        feature_importance['Importance'] = feature_importance['Importance'] / feature_importance['Importance'].sum()
        return PermutationFeatureImportance(feature_importances=feature_importance.sort_values('Importance', ascending=False))
