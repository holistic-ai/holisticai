from __future__ import annotations

from typing import Literal, Union

import numpy as np
from numpy.random import RandomState

from holisticai.utils.surrogate_models._trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

Surrogate = Union[DecisionTreeClassifier, DecisionTreeRegressor]
LearningTask = Literal["binary_classification", "multi_classification", "regression"]
SurrogateType = Literal["shallow_tree", "tree"]


def create_surrogate_model(X, y_pred, surrogate_type, learning_task="classification"):
    if learning_task == "classification":
        if len(np.unique(y_pred)) == 2:
            surrogate = BinaryClassificationSurrogate(X, y_pred=y_pred, model_type=surrogate_type)
        elif len(np.unique(y_pred)) > 2:
            surrogate = MultiClassificationSurrogate(X, y_pred=y_pred, model_type=surrogate_type)
        else:
            msg = "y_pred must have at least two unique values"
            raise ValueError(msg)
    elif learning_task == "regression":
        surrogate = RegressionSurrogate(X, y_pred=y_pred, model_type=surrogate_type)
    else:
        msg = f"Learning task {learning_task} not supported"
        raise ValueError(msg)
    return surrogate


class BinaryClassificationSurrogate:
    def __new__(cls, X, y_pred, model_type: SurrogateType = "shallow_tree", random_state: RandomState | None = None):
        if random_state is None:
            random_state = RandomState(42)

        if model_type in ["shallow_tree", "tree"]:
            surrogate = DecisionTreeClassifier(
                learning_task="binary_classification", model_type=model_type, random_state=random_state
            )
            return surrogate.fit(X=X, y=y_pred)
        msg = f"Surrogate type {model_type} not supported"
        raise ValueError(msg)


class MultiClassificationSurrogate:
    def __new__(cls, X, y_pred, model_type: SurrogateType = "shallow_tree", random_state: RandomState | None = None):
        if random_state is None:
            random_state = RandomState(42)

        if model_type in ["shallow_tree", "tree"]:
            surrogate = DecisionTreeClassifier(
                learning_task="multi_classification", model_type=model_type, random_state=random_state
            )
            return surrogate.fit(X=X, y=y_pred)
        msg = f"Surrogate type {model_type} not supported"
        raise ValueError(msg)


class RegressionSurrogate:
    def __new__(cls, X, y_pred, model_type: SurrogateType = "shallow_tree", random_state: RandomState | None = None):
        if random_state is None:
            random_state = RandomState(42)
        if model_type in ["shallow_tree", "tree"]:
            surrogate = DecisionTreeRegressor(
                learning_task="regression", model_type=model_type, random_state=random_state
            )
            return surrogate.fit(X=X, y=y_pred)
        msg = f"Surrogate type {model_type} not supported"
        raise ValueError(msg)


class ClusteringSurrogate:
    def __new__(cls, X, y_pred, model_type: SurrogateType = "shallow_tree", random_state: RandomState | None = None):
        if random_state is None:
            random_state = RandomState(42)

        if model_type in ["shallow_tree", "tree"]:
            surrogate = DecisionTreeClassifier(
                learning_task="clustering", model_type=model_type, random_state=random_state
            )
            return surrogate.fit(X=X, y=y_pred)
        msg = f"Surrogate type {model_type} not supported"
        raise ValueError(msg)


def get_features(surrogate):
    if hasattr(surrogate, "feature"):
        return surrogate.feature

    if isinstance(surrogate, (DecisionTreeClassifier, DecisionTreeRegressor)):
        assert surrogate.__is_fitted__, "Model not fitted"
        return surrogate._surrogate.tree_.feature  # noqa: SLF001
    msg = f"Surrogate type {type(surrogate)} not supported"
    raise ValueError(msg)


def get_number_of_rules(surrogate):
    if hasattr(surrogate, "n_leaves"):
        return surrogate.n_leaves

    if isinstance(surrogate, (DecisionTreeClassifier, DecisionTreeRegressor)):
        assert surrogate.__is_fitted__, "Model not fitted"
        return surrogate._surrogate.get_n_leaves()  # noqa: SLF001
    msg = f"Surrogate type {type(surrogate)} not supported"
    raise ValueError(msg)
