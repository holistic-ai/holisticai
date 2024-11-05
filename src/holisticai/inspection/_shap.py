from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from numpy.random import RandomState

from holisticai.datasets._dataset import Dataset
from holisticai.inspection import group_mask_samples_by_learning_task
from holisticai.utils import LocalImportances, ModelProxy


def compute_shap_feature_importance(
    proxy: ModelProxy,
    X: pd.DataFrame,
    max_samples: Union[int, None] = None,
    random_state: Union[RandomState, None] = None,
) -> LocalImportances:
    if random_state is None:
        random_state = RandomState(42)

    if max_samples is None:
        max_samples = -1

    y_ = pd.Series(proxy.predict(X))
    if proxy.learning_task == "binary_classification" and hasattr(proxy, "predict_proba"):
        y_proba_ = pd.Series(proxy.predict_proba(X)[:, 1])
        ds = Dataset(X=X, y=y_, y_proba=y_proba_)
    else:
        ds = Dataset(X=X, y=y_)

    ds["cond"] = group_mask_samples_by_learning_task(
        ds["y"],
        proxy.learning_task,
    )
    num_cond = ds["cond"].nunique()
    num_samples_by_cond = (max_samples + num_cond - 1) // num_cond
    if max_samples > 0:
        ds = ds.groupby("cond").sample(n=num_samples_by_cond, random_state=random_state)

    pfi = SHAPImportanceCalculator()
    pfi.initialize_explainer(X, proxy)
    data = pfi.compute_importances(ds)

    if "y_proba" in ds.features:
        local_importances = LocalImportances(data=data, cond=ds["cond"], metadata=ds[["y", "y_proba"]])
    else:
        local_importances = LocalImportances(data=data, cond=ds["cond"], metadata=ds[["y"]])
    return local_importances


class SHAPImportanceCalculator:
    importance_type: str = "local"

    def initialize_explainer(self, X: pd.DataFrame, proxy: ModelProxy):
        try:
            import shap  # type: ignore
        except ImportError:
            raise ImportError("SHAP is not installed. Please install it using 'pip install shap'") from None

        self.explainer = shap.Explainer(proxy.predict, X[:100])

    def compute_importances(self, ds: Dataset):
        X = pd.DataFrame(ds["X"].astype(np.float64))
        shap_values = self.explainer(X)

        data = np.abs(shap_values.values)
        return pd.DataFrame(data=data, columns=X.columns)
