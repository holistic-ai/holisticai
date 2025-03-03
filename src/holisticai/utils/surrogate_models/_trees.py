from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from holisticai.utils.obj_rep.object_repr import ReprObj
from holisticai.utils.surrogate_models._base import SurrogateBase
from numpy.random import RandomState
from sklearn.metrics import accuracy_score, mean_squared_error


def validate_input(X, y=None):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if y is None:
        return X
    return X, y


class TreeBase(SurrogateBase):
    _surrogate = None

    @property
    def feature(self):
        assert self._surrogate is not None, "Model not fitted"
        return self._surrogate.tree_.feature

    def get_n_leaves(self):
        assert self._surrogate is not None, "Model not fitted"
        return self._surrogate.get_n_leaves()

    @property
    def children_left(self):
        assert self._surrogate is not None, "Model not fitted"
        return self._surrogate.tree_.children_left

    @property
    def children_right(self):
        assert self._surrogate is not None, "Model not fitted"
        return self._surrogate.tree_.children_right

    @property
    def n_node_samples(self):
        assert self._surrogate is not None, "Model not fitted"
        return self._surrogate.tree_.n_node_samples

    @property
    def n_classes(self):
        assert self._surrogate is not None, "Model not fitted"
        return self._surrogate.tree_.n_classes

    @property
    def value(self):
        assert self._surrogate is not None, "Model not fitted"
        return self._surrogate.tree_.value


class OptimalTreeBase(TreeBase):
    def predict(self, X):
        assert self._surrogate is not None, "Model not fitted"

        X = validate_input(X)
        return self._surrogate.predict(X)


class DecisionTreeClassifier(OptimalTreeBase, ReprObj):
    name = "Shallow Decision Tree Classifier"

    def __init__(
        self,
        learning_task: Literal["binary_classification", "multi_classification", "clustering"],
        model_type: Literal["shallow_tree", "tree"] = "shallow_tree",
        random_state: Optional[RandomState] = None,
    ):
        self.random_state = random_state
        self.model_type = model_type
        super().__init__(learning_task=learning_task)

    def build(self, X, y):
        X, y = validate_input(X, y)

        best_tree = None
        from sklearn.ensemble import RandomForestClassifier

        if self.model_type == "shallow_tree":
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, max_depth=3)
        elif self.model_type == "tree":
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
        rf.fit(X, y)
        best_score = -np.inf
        for tree in rf.estimators_:
            predicciones = tree.predict(X)
            score = accuracy_score(y, predicciones)
            if score > best_score:
                best_score = score
                best_tree = tree
        return best_tree

    @property
    def feature_importances_(self):
        assert self._surrogate is not None, "Model not fitted"
        return self._surrogate.feature_importances_

    def repr_info(self):
        return {
            "dtype": "Surrogate Model",
            "attributes": {
                "Learning Task": self.learning_task,
                "Name": self.name,
                "Model Type": self.model_type,
            },
        }


class DecisionTreeRegressor(OptimalTreeBase, ReprObj):
    name = "Shallow Decision Tree Regressor"

    def __init__(
        self,
        learning_task: Literal["regression"],
        model_type: Literal["shallow_tree", "tree"] = "shallow_tree",
        random_state: Optional[RandomState] = None,
    ):
        self.model_type = model_type
        self.random_state = random_state
        super().__init__(learning_task=learning_task)

    def build(self, X, y):
        X, y = validate_input(X, y)

        best_tree = None
        from sklearn.ensemble import RandomForestRegressor

        if self.model_type == "shallow_tree":
            rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state, max_depth=3)
        elif self.model_type == "tree":
            rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
        rf.fit(X, y)
        best_score = np.inf
        for tree in rf.estimators_:
            predicciones = tree.predict(X)
            score = mean_squared_error(y, predicciones)
            if score < best_score:
                best_score = score
                best_tree = tree
        return best_tree

    @property
    def feature_importances_(self):
        assert self._surrogate is not None, "Model not fitted"
        return self._surrogate.feature_importances_

    def repr_info(self):
        return {
            "dtype": "Surrogate Model",
            "attributes": {
                "Learning Task": self.learning_task,
                "Name": self.name,
                "Model Type": self.model_type,
            },
        }
