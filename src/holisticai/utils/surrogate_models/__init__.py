from __future__ import annotations

from typing import Literal, Optional, Union

from numpy.random import RandomState

from holisticai.utils.surrogate_models._trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

SklearnDecisionTree = Union[DecisionTreeClassifier, DecisionTreeRegressor]

Surrogate = SklearnDecisionTree
LearningTask = Literal["binary_classification", "multi_classification", "regression"]
SurrogateType = Literal["shallow_tree", "tree"]


class BinaryClassificationSurrogate:
    def __new__(cls, X, y_pred, model_type: SurrogateType = "shallow_tree", random_state: Optional[RandomState] = None):
        if random_state is None:
            random_state = RandomState(42)

        if model_type in ["shallow_tree", "tree"]:
            surrogate = DecisionTreeClassifier(
                learning_task="binary_classification", model_type=model_type, random_state=random_state
            )
            return surrogate.fit(X=X, y=y_pred)
        raise ValueError(f"Surrogate type {model_type} not supported")


class MultiClassificationSurrogate:
    def __new__(cls, X, y_pred, model_type: SurrogateType = "shallow_tree", random_state: Optional[RandomState] = None):
        if random_state is None:
            random_state = RandomState(42)

        if model_type in ["shallow_tree", "tree"]:
            surrogate = DecisionTreeClassifier(
                learning_task="multi_classification", model_type=model_type, random_state=random_state
            )
            return surrogate.fit(X=X, y=y_pred)
        raise ValueError(f"Surrogate type {model_type} not supported")


class RegressionSurrogate:
    def __new__(cls, X, y_pred, model_type: SurrogateType = "shallow_tree", random_state: Optional[RandomState] = None):
        if random_state is None:
            random_state = RandomState(42)
        if model_type in ["shallow_tree", "tree"]:
            surrogate = DecisionTreeRegressor(
                learning_task="regression", model_type=model_type, random_state=random_state
            )
            return surrogate.fit(X=X, y=y_pred)
        raise ValueError(f"Surrogate type {model_type} not supported")


class ClusteringSurrogate:
    def __new__(cls, X, y_pred, model_type: SurrogateType = "shallow_tree", random_state: Optional[RandomState] = None):
        if random_state is None:
            random_state = RandomState(42)

        if model_type in ["shallow_tree", "tree"]:
            surrogate = DecisionTreeClassifier(
                learning_task="clustering", model_type=model_type, random_state=random_state
            )
            return surrogate.fit(X=X, y=y_pred)
        raise ValueError(f"Surrogate type {model_type} not supported")


def get_features(surrogate):
    if hasattr(surrogate, "feature"):
        return surrogate.feature

    if isinstance(surrogate, SklearnDecisionTree):
        assert surrogate.__is_fitted__, "Model not fitted"
        return surrogate._surrogate.tree_.feature  # noqa: SLF001
    raise ValueError(f"Surrogate type {type(surrogate)} not supported")


def get_number_of_rules(surrogate):
    if hasattr(surrogate, "n_leaves"):
        return surrogate.n_leaves

    if isinstance(surrogate, SklearnDecisionTree):
        assert surrogate.__is_fitted__, "Model not fitted"
        return surrogate._surrogate.get_n_leaves()  # noqa: SLF001
    raise ValueError(f"Surrogate type {type(surrogate)} not supported")
