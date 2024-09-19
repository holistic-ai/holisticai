import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.metrics import accuracy_score, mean_squared_error

from holisticai.utils.models.surrogate._base import SurrogateBase


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


class ShallowDecisionTreeClassifier(OptimalTreeBase):
    name = "Shallow Decision Tree Classifier"
    def __init__(self, random_state: RandomState):
        self.random_state = random_state
        super().__init__()

    def build(self, X, y):
        X,y = validate_input(X, y)

        best_tree = None
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, max_depth=3)
        rf.fit(X, y)
        best_score = -np.inf
        for tree in rf.estimators_:
            predicciones = tree.predict(X)
            score = accuracy_score(y, predicciones)
            if score > best_score:
                best_score = score
                best_tree = tree
        return best_tree


class ShallowDecisionTreeRegressor(OptimalTreeBase):
    name = "Shallow Decision Tree Regressor"
    def __init__(self, random_state: RandomState):
        self.random_state = random_state
        super().__init__()


    def build(self, X, y):
        X,y = validate_input(X, y)

        best_tree = None
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state, max_depth=3)
        rf.fit(X, y)
        best_score = np.inf
        for tree in rf.estimators_:
            predicciones = tree.predict(X)
            score = mean_squared_error(y, predicciones)
            if score < best_score:
                best_score = score
                best_tree = tree
        return best_tree


class DecisionTreeClassifier(OptimalTreeBase):
    name = "Decision Tree Classifier"
    def __init__(self, random_state: RandomState):
        self.random_state = random_state
        super().__init__()

    def build(self, X, y):
        X,y = validate_input(X, y)

        best_tree = None
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X, y)
        best_score = -np.inf
        for tree in rf.estimators_:
            predicciones = tree.predict(X)
            score = accuracy_score(y, predicciones)
            if score > best_score:
                best_score = score
                best_tree = tree
        return best_tree


class DecisionTreeRegressor(OptimalTreeBase):
    name = "Decision Tree Regressor"
    def __init__(self, random_state: RandomState):
        self.random_state = random_state
        super().__init__()


    def build(self, X, y):
        X,y = validate_input(X, y)

        best_tree = None
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        rf.fit(X, y)
        best_score = np.inf
        for tree in rf.estimators_:
            predicciones = tree.predict(X)
            score = mean_squared_error(y, predicciones)
            if score < best_score:
                best_score = score
                best_tree = tree
        return best_tree
