from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from numpy.random import RandomState

from holisticai.datasets import Dataset
from holisticai.utils import LocalImportances, ModelProxy
from holisticai.utils.feature_importances import group_samples_by_learning_task


def compute_shap_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    proxy: ModelProxy,
    max_samples: Union[int, None] = None,
    random_state: Union[RandomState, None] = None,
) -> LocalImportances:
    if random_state is None:
        random_state = RandomState(42)

    if max_samples is None:
        max_samples = -1

    ds = Dataset(X=X, y=y)
    if max_samples > 0:
        ds = ds.sample(n=max_samples, random_state=random_state)

    pfi = SHAPImportanceCalculator()
    data = pfi.compute_importances(ds, proxy)
    condition = group_samples_by_learning_task(ds["y"], proxy.learning_task, return_group_mask=True)
    local_importances = LocalImportances(data=data, cond=condition)
    return local_importances


class SHAPImportanceCalculator:
    importance_type: str = "local"

    def initialize_explainer(self, X: pd.DataFrame, proxy: ModelProxy):
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP is not installed. Please install it using 'pip install shap'") from None

        masker = shap.maskers.Independent(X)
        self.explainer = shap.Explainer(proxy.predict, masker=masker)

    def compute_importances(self, ds: Dataset, proxy: ModelProxy):
        X = ds["X"].astype(np.float64)
        self.initialize_explainer(X, proxy)
        shap_values = self.explainer(X)

        data = np.abs(shap_values.values)
        data = data / data.sum(axis=1)[:, None]
        return pd.DataFrame(data=data, columns=X.columns)
