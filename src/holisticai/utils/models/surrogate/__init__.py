from __future__ import annotations

from typing import Literal, Optional, Union

from numpy.random import RandomState

from holisticai.utils.models.surrogate._trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ShallowDecisionTreeClassifier,
    ShallowDecisionTreeRegressor,
)

SklearnDecisionTree = Union[ShallowDecisionTreeClassifier, ShallowDecisionTreeRegressor, DecisionTreeClassifier, DecisionTreeRegressor]
#TODO add other surrogate models
Surrogate = SklearnDecisionTree
LearningTask = Literal["binary_classification", "multi_classification", "regression"]
SurrogateType = Literal["shallow_tree", "tree"]


def SurrogateModel(proxy, X, model_type:SurrogateType="shallow_tree", random_state: Optional[RandomState]=None) -> Surrogate:  # noqa: N802
    if random_state is None:
        random_state = RandomState(42)

    if model_type == "shallow_tree":
        if proxy.learning_task in ["binary_classification", "multi_classification"]:
            surrogate = ShallowDecisionTreeClassifier(random_state=random_state)
        elif proxy.learning_task == "regression":
            surrogate = ShallowDecisionTreeRegressor(random_state=random_state)
        else:
            raise ValueError(f"Learning task {proxy.learning_task} not supported")
        return surrogate.fit(X=X, y=proxy.predict(X))

    if model_type == "tree":
        if proxy.learning_task in ["binary_classification", "multi_classification"]:
            surrogate = DecisionTreeClassifier(random_state=random_state)
        elif proxy.learning_task == "regression":
            surrogate = DecisionTreeRegressor(random_state=random_state)
        else:
            raise ValueError(f"Learning task {proxy.learning_task} not supported")
        return surrogate.fit(X=X, y=proxy.predict(X))

    raise ValueError(f"Surrogate type {model_type} not supported")

def get_features(surrogate):
    if isinstance(surrogate, SklearnDecisionTree):
        assert surrogate.__is_fitted__, "Model not fitted"
        return surrogate._surrogate.tree_.feature  # noqa: SLF001
    raise ValueError(f"Surrogate type {type(surrogate)} not supported")

def get_number_of_rules(surrogate):
    if isinstance(surrogate, SklearnDecisionTree):
        assert surrogate.__is_fitted__, "Model not fitted"
        return surrogate._surrogate.get_n_leaves()  # noqa: SLF001
    raise ValueError(f"Surrogate type {type(surrogate)} not supported")

def get_feature_importances(surrogate):
    if isinstance(surrogate, SklearnDecisionTree):
        assert surrogate.__is_fitted__, "Model not fitted"
        return surrogate._surrogate.feature_importances_  # noqa: SLF001
    raise ValueError(f"Surrogate type {type(surrogate)} not supported")

