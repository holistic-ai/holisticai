from typing import Optional

import numpy as np
import pandas as pd

from holisticai.utils.transformers.bias import BMPreprocessing as BMPre
from holisticai.utils.transformers.bias import SensitiveGroups


class Reweighing(BMPre):
    """
    Reweighing preprocessing weights the examples in each group-label combination to ensure fairness before
    classification.

    References
    ----------
        Kamiran, Faisal, and Toon Calders. "Data preprocessing techniques for classification
        without discrimination." Knowledge and information systems 33.1 (2012): 1-33.
    """

    def __init__(self):
        self.sens_groups = SensitiveGroups()

    def fit(
        self,
        y_true: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Fit.

        Description
        ----------
        Access fitted sample_weight param with self.estimator_params["sample_weight"].

        Parameters
        ----------
        y_true : array-like
            Target vector
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        sample_weight (optional) : array-like
            Samples weights vector

        Returns
        -------
        Self
        """

        params = self._load_data(
            y_true=y_true, sample_weight=sample_weight, group_a=group_a, group_b=group_b
        )
        y_true = params["y_true"]
        sample_weight = params["sample_weight"]
        group_a = params["group_a"]
        group_b = params["group_b"]

        group_lbs = self.sens_groups.fit_transform(np.stack([group_a, group_b], axis=1))

        classes = np.unique(y_true)

        df = pd.DataFrame()

        df["LABEL"] = pd.Series(y_true)

        df["GROUP_ID"] = group_lbs

        df["COUNT"] = 1

        for g in self.sens_groups.group_names:
            for c in classes:
                df[f"{g}-{c}"] = (df["GROUP_ID"] == g) & (df["LABEL"] == c)

        df_group_values = df.groupby(["GROUP_ID", "LABEL"])["COUNT"].sum()

        df_values = df_group_values.groupby(level="LABEL").sum()

        df_groups = df_group_values.groupby(level="GROUP_ID").sum()

        df_group_values_prob = df_group_values / df_groups

        df_values_prob = df_values / df_values.sum()

        df_group_values_weights = df_values_prob / df_group_values_prob

        sample_weight = np.ones_like(y_true, dtype=np.float32)
        for g in self.sens_groups.group_names:
            for l in classes:
                mask = df[f"{g}-{l}"]
                sample_weight[mask] = df_group_values_weights.at[g, l]

        self.update_estimator_param("sample_weight", sample_weight)

        return self

    def transform(self, X: np.ndarray):
        """passthrough"""
        return X

    def fit_transform(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Fit transform.

        Description
        ----------
        Access fitted sample_weight param with self.estimator_params["sample_weight"].
        The transform returns the same object inputed.

        Parameters
        ----------
        X : matrix-like
            Input matrix
        y_true : array-like
            Target vector
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)
        sample_weight (optional) : array-like
            Samples weights vector

        Returns
        -------
            X
        """
        return self.fit(y_true, group_a, group_b, sample_weight).transform(X)
