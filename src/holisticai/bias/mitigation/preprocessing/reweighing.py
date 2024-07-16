from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from holisticai.utils.transformers.bias import BMPreprocessing as BMPre
from holisticai.utils.transformers.bias import SensitiveGroups


class Reweighing(BMPre):
    """
    Reweighing preprocessing weights the examples in each group-label combination to ensure fairness before\
    classification.

    References
    ----------
    .. [1] Kamiran, Faisal, and Toon Calders. "Data preprocessing techniques for classification\
        without discrimination." Knowledge and information systems 33.1 (2012): 1-33.
    """

    def __init__(self):
        self._sensgroups = SensitiveGroups()

    def fit(
        self,
        y: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Fit the Reweighing model to the data. This method calculates the sample weights to ensure that the \
        data is fair with respect to the specified sensitive groups before classification.

        Parameters
        ----------
        y : array-like
            Target vector
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        sample_weight  : array-like, optional
            Samples weights vector. Default is None.

        Returns
        -------
        Self
        """

        params = self._load_data(y=y, sample_weight=sample_weight, group_a=group_a, group_b=group_b)
        y = params["y"]
        sample_weight = params["sample_weight"]
        group_a = params["group_a"]
        group_b = params["group_b"]

        group_lbs = self._sensgroups.fit_transform(np.stack([group_a, group_b], axis=1))

        classes = np.unique(y)

        df = pd.DataFrame()

        df["LABEL"] = pd.Series(y)

        df["GROUP_ID"] = group_lbs

        df["COUNT"] = 1

        for g in self._sensgroups.group_names:
            for c in classes:
                df[f"{g}-{c}"] = (df["GROUP_ID"] == g) & (df["LABEL"] == c)

        df_group_values = df.groupby(["GROUP_ID", "LABEL"])["COUNT"].sum()

        df_values = df_group_values.groupby(level="LABEL").sum()

        df_groups = df_group_values.groupby(level="GROUP_ID").sum()

        df_group_values_prob = df_group_values / df_groups

        df_values_prob = df_values / df_values.sum()

        df_group_values_weights = df_values_prob / df_group_values_prob

        self.sample_weight = np.ones_like(y, dtype=np.float32)
        for g in self._sensgroups.group_names:
            for c in classes:
                mask = df[f"{g}-{c}"]
                self.sample_weight[mask] = df_group_values_weights.at[g, c]

        self._update_estimator_param("sample_weight", self.sample_weight)

        return self

    def transform(self, X: np.ndarray):
        """passthrough"""
        return X

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Fit the Reweighing model to the data. This method calculates the sample weights to ensure that the \
        data is fair with respect to the specified sensitive groups before classification.
        The transform function returns the same object inputed.

        Parameters
        ----------
        X : matrix-like
            Input matrix
        y : array-like
            Target vector
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        sample_weight : array-like, optional
            Samples weights vector. Default is None.

        Returns
        -------
        self
        """
        return self.fit(y, group_a, group_b, sample_weight).transform(X)
