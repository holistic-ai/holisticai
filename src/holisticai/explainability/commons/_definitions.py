from __future__ import annotations

from typing import Annotated, Any, Literal, Union

import pandas as pd  # noqa: TCH002
from pydantic import BaseModel, ConfigDict, Field


class BinaryClassificationXAISettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    learning_task: Literal["binary_classification"] = "binary_classification"
    predict_fn: callable
    predict_proba_fn: callable
    classes: list


class MultiClassificationXAISettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    learning_task: Literal["multi_classification"] = "multi_classification"
    predict_fn: callable
    predict_proba_fn: callable
    classes: list


class RegressionClassificationXAISettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    learning_task: Literal["regression"] = "regression"
    predict_fn: callable


LearningTaskXAISettings = Annotated[
    Union[BinaryClassificationXAISettings, MultiClassificationXAISettings, RegressionClassificationXAISettings],
    Field(discriminator="learning_task"),
]


class Importances(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    feature_importances: pd.DataFrame

    @property
    def feature_names(self):
        return self.feature_importances.Variable.tolist()

    def __len__(self):
        return len(self.feature_importances)


class PermutationFeatureImportance(Importances):
    strategy: Literal["permutation"] = "permutation"


class SurrogateFeatureImportance(Importances):
    surrogate: Any
    strategy: Literal["surrogate"] = "surrogate"


FeatureImportance = Annotated[
    Union[PermutationFeatureImportance, SurrogateFeatureImportance], Field(discriminator="strategy")
]


class ConditionalFeatureImportance(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    conditional_feature_importance: dict[str, Importances]

    @property
    def feature_names(self):
        return {
            name: self.feature_importances.Variable.tolist() for name in self.conditional_feature_importance.items()
        }


class LocalImportances(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    feature_importances: pd.DataFrame

    @property
    def feature_names(self):
        return list(self.feature_importances.columns)


class LocalConditionalFeatureImportance(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    conditional_feature_importance: dict[str, LocalImportances]

    @property
    def feature_names(self):
        return {
            name: self.feature_importances.Variable.tolist() for name in self.conditional_feature_importance.items()
        }

class PartialDependence(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    partial_dependence: list[dict]
