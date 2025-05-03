from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from numpy.random import RandomState

from holisticai.datasets._dataset import Dataset
from holisticai.inspection._utils import group_mask_samples_by_learning_task
from holisticai.utils._definitions import LocalImportances, ModelProxy

warnings.filterwarnings("ignore")


def compute_lime_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    proxy: ModelProxy,
    max_samples: int | None = None,
    random_state: RandomState | None = None,
) -> LocalImportances:
    if random_state is None:
        random_state = RandomState(42)

    if max_samples is None:
        max_samples = -1

    ds = Dataset(X=X, y=y)
    if max_samples > 0:
        ds = ds.sample(n=max_samples, random_state=random_state)

    pfi = LIMEImportanceCalculator()
    data = pfi.compute_importances(ds, proxy)
    # y_: pd.Series = pd.Series(ds["y"])
    y_ = pd.Series(proxy.predict(ds["X"]))
    condition = group_mask_samples_by_learning_task(y_, proxy.learning_task)
    return LocalImportances(data=data, cond=condition)


class LIMEImportanceCalculator:
    importance_type: str = "local"

    def initialize_explainer(self, X: pd.DataFrame, proxy: ModelProxy):
        try:
            import lime  # type: ignore
            import lime.lime_tabular  # type: ignore
        except ImportError:
            msg = "LIME is not installed. Please install it using 'pip install lime'"
            raise ImportError(msg) from None

        if proxy.learning_task == "regression":
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                X.values,
                feature_names=X.columns,
                discretize_continuous=True,
                mode="regression",
                random_state=42,
            )
            self.predict_function = proxy.predict
            self.feature_names = list(X.columns)
        elif proxy.learning_task in ["binary_classification", "multi_classification"]:
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                X.values,
                feature_names=X.columns,
                class_names=proxy.classes,
                discretize_continuous=True,
                mode="classification",
                random_state=42,
            )
            self.predict_function = proxy.predict_proba
            self.feature_names = list(X.columns)
        else:
            msg = "Learning task must be regression or classification"
            raise ValueError(msg)

    def compute_importances(self, ds: Dataset, proxy: ModelProxy) -> pd.DataFrame:
        X = pd.DataFrame(ds["X"].astype(np.float64))
        self.initialize_explainer(X, proxy)
        importances = []
        for i in range(len(ds)):
            instance = ds["X"].iloc[i].values.reshape(1, -1)
            exp = self.explainer.explain_instance(
                instance[0], self.predict_function, num_features=len(self.feature_names)
            )
            exp_values = np.array(next(iter(exp.local_exp.values())))
            ranked_imp = exp_values[:, 1]
            rank = exp_values[:, 0].argsort()
            importance = np.zeros_like(ranked_imp)
            importance[rank] = ranked_imp
            importances.append(importance)
        importances = np.stack(importances)
        data = np.abs(importances)
        data = data / data.sum(axis=1)[:, None]
        return pd.DataFrame(data=data, columns=ds["X"].columns)
