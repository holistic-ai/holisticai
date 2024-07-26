import logging

import numpy as np
import pandas as pd
from holisticai.bias.mitigation.inprocessing.variational_fair_clustering.algorithm_utils._fair_clustering import (
    FairClustering,
)
from holisticai.bias.mitigation.inprocessing.variational_fair_clustering.algorithm_utils._utils import (
    normalizefea,
)


class FairClusteringAlgorithm:
    def __init__(self, K, L, lmbda, method="kmeans", normalize_input=True, verbose=True):
        self.lmbda = lmbda
        self.K = K
        self.L = L
        self.normalize_input = normalize_input
        self.fair_clustering = FairClustering(K, L, lmbda, method=method, verbose=verbose)

    def fit_transform_normalize(self, X):
        self.meanX = np.nanmean(X, axis=0)
        self.stdX = np.nanstd(X, axis=0)
        if not all(list(self.stdX > 1e-15)):
            logging.warning(f"Some features have std=0, {self.stdX}")
        self.stdX = np.where(self.stdX < 1e-15, 1, self.stdX)
        return self.transform_normalize(X)

    def transform_normalize(self, X):
        return (X - self.meanX) / self.stdX

    def _preprocess_data(self, p_attr):
        groups_ids = pd.DataFrame()
        p_attr = pd.Series(p_attr)

        for g in np.unique(p_attr):
            groups_ids[f"{g}"] = p_attr == g

        group_count = p_attr.value_counts()
        group_prob = group_count / len(p_attr)

        return group_prob, groups_ids

    def fit(self, X, p_attr, random_state):
        group_prob, groups_ids = self._preprocess_data(p_attr)
        # Scale and Normalize Features
        if self.normalize_input:
            X = self.fit_transform_normalize(X)
            X = normalizefea(X)

        self.fair_clustering.fit(X, group_prob, groups_ids, random_state)

        return self

    def predict(self, X, p_attr):
        _, groups_ids = self._preprocess_data(p_attr)
        if self.normalize_input:
            X = self.transform_normalize(X)
            X = normalizefea(X)
        return self.fair_clustering.predict(X, groups_ids)
