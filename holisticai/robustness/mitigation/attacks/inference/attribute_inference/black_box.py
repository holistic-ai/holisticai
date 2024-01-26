"""
This module implements attribute inference attacks.
"""
import logging
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
from sklearn.preprocessing import minmax_scale

from holisticai.robustness.mitigation.attacks import AttributeInferenceAttack
from holisticai.robustness.mitigation.utils.formatting import (
    check_and_transform_label_format,
    float_to_categorical,
    floats_to_one_hot,
    get_feature_index,
    get_feature_values,
)
from holisticai.robustness.utils import AttributeInferenceDataPreprocessor
from holisticai.wrappers.classification.classifier import ClassifierMixin
from holisticai.wrappers.estimator import BaseEstimator
from holisticai.wrappers.regression.regressor import RegressorMixin

from .utils import get_attack_model

if TYPE_CHECKING:
    from holisticai.wrappers.formatting import CLASSIFIER_TYPE, REGRESSOR_TYPE

logger = logging.getLogger(__name__)


class AttributeInferenceBlackBox(AttributeInferenceAttack):
    """
    Implementation of a simple black-box attribute inference attack.

    The idea is to train a simple neural network to learn the attacked feature from the rest of the features and the
    model's predictions. Assumes the availability of the attacked model's predictions for the samples under attack,
    in addition to the rest of the feature values. If this is not available, the true class label of the samples may be
    used as a proxy.
    """

    attack_params = AttributeInferenceAttack.attack_params + [
        "prediction_normal_factor",
        "scale_range",
        "attack_model_type",
    ]
    _estimator_requirements = (BaseEstimator, (ClassifierMixin, RegressorMixin))

    def __init__(
        self,
        estimator: Union["CLASSIFIER_TYPE", "REGRESSOR_TYPE"],
        attack_model_type: str = "nn",
        attack_model: Optional["CLASSIFIER_TYPE"] = None,
        attack_feature: Union[int, slice] = 0,
        scale_range: Optional[Tuple[float, float]] = None,
        prediction_normal_factor: Optional[float] = 1,
    ):
        """
        Create an AttributeInferenceBlackBox attack instance.

        :param estimator: Target estimator.
        :param attack_model_type: the type of default attack model to train, optional. Should be one of `nn` (for neural
                                  network, default) or `rf` (for random forest). If `attack_model` is supplied, this
                                  option will be ignored.
        :param attack_model: The attack model to train, optional. If none is provided, a default model will be created.
        :param attack_feature: The index of the feature to be attacked or a slice representing multiple indexes in
                               case of a one-hot encoded feature.
        :param scale_range: If supplied, the class labels (both true and predicted) will be scaled to the given range.
                            Only applicable when `estimator` is a regressor.
        :param prediction_normal_factor: If supplied, the class labels (both true and predicted) are multiplied by the
                                         factor when used as inputs to the attack-model. Only applicable when
                                         `estimator` is a regressor and if `scale_range` is not supplied.
        """
        super().__init__(estimator=estimator, attack_feature=attack_feature)

        self._values: Optional[list] = None
        self._nb_classes: Optional[int] = None
        self._attack_model_type = attack_model_type
        self._attack_model = attack_model

        if attack_model:
            if ClassifierMixin not in type(attack_model).__mro__:
                raise ValueError("Attack model must be of type Classifier.")
            self.attack_model = attack_model
        else:
            self.attack_model = get_attack_model(attack_model_type)

        self.prediction_normal_factor = prediction_normal_factor
        self.scale_range = scale_range
        is_regression = not (ClassifierMixin in type(self.estimator).__mro__)
        self._check_params()
        self.attack_feature = get_feature_index(self.attack_feature)
        self.ai_preprocessor = AttributeInferenceDataPreprocessor(
            is_regression=is_regression,
            scale_range=scale_range,
            prediction_normal_factor=prediction_normal_factor,
            attack_feature=attack_feature,
        )

    def fit(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        pred: Optional[np.ndarray] = None,
    ) -> None:
        """
        Train the attack model.

        :param x: Input to training process. Includes all features used to train the original model.
        :param y: True labels for x.
        """

        # train attack model
        attack_x, attack_y = self.ai_preprocessor.fit_transform(x, y, pred)
        self._values = self.ai_preprocessor._values
        self.attack_model.fit(attack_x, attack_y)

    def infer(
        self, x: np.ndarray, y: np.ndarray, pred: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Infer the attacked feature.

        :param x: Input to attack. Includes all features except the attacked feature.
        :param y: True labels for x.
        :param pred: Original model's predictions for x.
        :type pred: `np.ndarray`
        :param values: Possible values for attacked feature. For a single column feature this should be a simple list
                       containing all possible values, in increasing order (the smallest value in the 0 index and so
                       on). For a multi-column feature (for example 1-hot encoded and then scaled), this should be a
                       list of lists, where each internal list represents a column (in increasing order) and the values
                       represent the possible values for that column (in increasing order). If not provided, is
                       computed from the training data when calling `fit`.
        :type values: list, optional
        :return: The inferred feature values.
        """

        values: Optional[list] = kwargs.get("values")

        # if provided, override the values computed in fit()
        if values is not None:
            self._values = values

        attack_x = self.ai_preprocessor.transform(x, y, pred)
        predictions = self.attack_model.predict_proba(attack_x).astype(np.float32)

        if self._values is not None:
            if isinstance(self.attack_feature, int):
                predictions = np.array(
                    [self._values[np.argmax(arr)] for arr in predictions]
                )
            else:
                i = 0
                for column in predictions.T:
                    for index in range(len(self._values[i])):
                        np.place(column, [column == index], self._values[i][index])
                    i += 1
        return np.array(predictions)

    def _check_params(self) -> None:

        if not isinstance(self.attack_feature, int) and not isinstance(
            self.attack_feature, slice
        ):
            raise ValueError(
                "Attack feature must be either an integer or a slice object."
            )

        if isinstance(self.attack_feature, int) and self.attack_feature < 0:
            raise ValueError("Attack feature index must be positive.")

        if self._attack_model_type not in ["nn", "rf"]:
            raise ValueError("Illegal value for parameter `attack_model_type`.")

        if RegressorMixin not in type(self.estimator).__mro__:
            if self.prediction_normal_factor != 1:
                raise ValueError(
                    "Prediction normal factor is only applicable to regressor models."
                )
