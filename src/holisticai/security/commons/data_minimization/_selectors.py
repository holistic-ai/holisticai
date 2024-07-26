from __future__ import annotations

import logging
from typing import Literal, get_args

import numpy as np
import pandas as pd
from holisticai.utils import Importances, ModelProxy
from holisticai.utils.feature_importances import compute_permutation_feature_importance
from sklearn.feature_selection import (
    SelectPercentile,
    VarianceThreshold,
    f_classif,
    f_regression,
)

logger = logging.getLogger(__name__)
SelectorbyData = Literal["Percentile", "Variance"]
SelectorbyImportance = Literal["FImportance"]
SelectorType = Literal[SelectorbyData, SelectorbyImportance]


def get_score_fn(learning_task: str):
    if learning_task in ["binary_classification", "multi_classification"]:
        return f_classif
    if learning_task == "regression":
        return f_regression
    raise ValueError(f"Learning task {learning_task} not supported")


class SelectorsHandler:
    def __init__(self, proxy: ModelProxy, selector_types: list[SelectorType]):
        self.proxy = proxy
        self.percentiles = [80, 90]
        self.variance_thresholds = [80, 90]
        self.importance_thresholds = [80, 90]
        self.selector_types = selector_types
        self.selectors = {}

    def _get_selectors_from_data(self, selector_type: SelectorbyData):
        percent_range = np.array(self.percentiles)
        variance_thresholds = np.array(self.variance_thresholds)

        if selector_type == "Percentile":
            methods = {}
            for p in percent_range:
                score_fn = get_score_fn(self.proxy.learning_task)
                methods[f"Percentile >{p}"] = SelectPercentile(score_fn, percentile=p)
            return methods

        if selector_type == "Variance":
            methods = {}
            for p in variance_thresholds:
                methods[f"Variance >{p}"] = VarianceThreshold(threshold=p / 100)
            return methods
        raise ValueError(f"Selector type {selector_type} not supported")

    def _get_selectors_from_importances(self, selector_type: SelectorbyImportance):
        if selector_type == "FImportance":
            methods = {}
            for p in self.importance_thresholds:
                methods[f"FImportance >{p}"] = SelectFromFeatureImportance(threshold=p / 100)
            return methods
        raise ValueError(f"Selector type {selector_type} not supported")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        selectors_by_data = {}
        selectors_by_importance = {}

        importances = None
        for selector_type in self.selector_types:
            if selector_type in get_args(SelectorbyData):
                selectors_by_data.update(self._get_selectors_from_data(selector_type))

            if selector_type in get_args(SelectorbyImportance):
                selectors_by_importance.update(self._get_selectors_from_importances(selector_type))

        for sn, selector in selectors_by_data.items():
            logger.info(f"Fitting selector {sn}")
            Xt = (
                X.select_dtypes(include=["number"])
                if any(sn.startswith(name) for name in ["Percentile", "Variance"])
                else X
            )
            selector.fit(Xt, y)

        for sn, selector in selectors_by_importance.items():
            logger.info(f"Fitting selector {sn}")
            if importances is None:
                importances = compute_permutation_feature_importance(proxy=self.proxy, X=X, y=y)
            selector.fit(importances)

        self.selectors = {**selectors_by_data, **selectors_by_importance}
        return self

    def __iter__(self):
        return iter(self.selectors.items())

    def get_feature_indexes(self):
        return {f: s.get_feature_names_out() for f, s in self.selectors.items()}


class SelectFromFeatureImportance:
    def __init__(self, threshold: float):
        """
        Parameters
        ----------
        feat_imp : FeatureImportance
            An FeatureImportance instance.

        threshold : float
            Used to select the minimum amount of features that cover this importance fraction.
        """
        self.threshold = threshold

    def fit(self, importances: Importances):
        """
        Create a list of top features based on the threshold.
        Return
        ------
            Self
        """
        self._importances = importances
        top_importances = self._importances.top_alpha(self.threshold)
        self.top_feature_names = top_importances.feature_names
        self.top_feature_index = [
            int(i) for i, fi in enumerate(self._importances.feature_names) if fi in self.top_feature_names
        ]
        return self

    def transform(self, X: pd.DataFrame):
        """
        Parameters
        ----------
        x: matrix-like
            input data (n_examples, n_features)

        Return
        ------
            matrix-like
            x matrix with only the selected features.
        """
        return X[self.top_feature_names]

    def get_feature_names_out(self):
        "Return list of columns index"
        return self.top_feature_index

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        return self.fit(X, y).transform(X)
