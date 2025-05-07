"""
This module implements attribute inference attacks.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np

from holisticai.security.attackers.attribute_inference.dataset_utils import AttributeInferenceDataPreprocessor
from holisticai.security.attackers.attribute_inference.utils import get_attack_model, get_feature_index

logger = logging.getLogger(__name__)


class AttributeInferenceBlackBox:
    """
    Implementation of a simple black-box attribute inference attack.

    The idea is to train a simple neural network to learn the attacked feature from the rest of the features and the
    model's predictions. Assumes the availability of the attacked model's predictions for the samples under attack,
    in addition to the rest of the feature values. If this is not available, the true class label of the samples may be
    used as a proxy.

    Parameters
    ----------
    estimator : object
        Target estimator.
    attack_model_type : str
        The type of default attack model to train, optional. Should be one of `nn` (for neural network, default) or `rf`
        (for random forest). If `attack_model` is supplied, this option will be ignored.
    attack_model : object
        The attack model to train, optional. If none is provided, a default model will be created.
    attack_feature : int or slice
        The index of the feature to be attacked or a slice representing multiple indexes in case of a one-hot encoded
        feature.
    scale_range : tuple
        If supplied, the class labels (both true and predicted) will be scaled to the given range. Only applicable when
        `estimator` is a regressor.
    prediction_normal_factor : float
        If supplied, the class labels (both true and predicted) are multiplied by the factor when used as inputs to the
        attack-model. Only applicable when `estimator` is a regressor and if `scale_range` is not supplied.
    """

    def __init__(
        self,
        estimator,
        attack_model_type: str = "nn",
        attack_model=None,
        attack_feature: Union[int, slice] = 0,
        scale_range: Optional[tuple[float, float]] = None,
        prediction_normal_factor: Optional[float] = 1,
    ):
        self._values: Optional[list] = None
        self._nb_classes: Optional[int] = None
        self._attack_model_type = attack_model_type
        self._attack_model = attack_model
        self.estimator = estimator
        self.attack_feature = attack_feature
        self.attack_model = get_attack_model(attack_model_type)

        self.prediction_normal_factor = prediction_normal_factor
        self.scale_range = scale_range
        self._check_params()
        self.attack_feature = get_feature_index(self.attack_feature)
        self.ai_preprocessor = AttributeInferenceDataPreprocessor(
            scale_range=scale_range,
            prediction_normal_factor=prediction_normal_factor,
            attack_feature=attack_feature,
        )

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, pred: Optional[np.ndarray] = None) -> None:
        """
        Train the attack model.

        Parameters
        ----------
        x : np.ndarray
            Input to training process. Includes all features used to train the original model.
        y : np.ndarray
            True labels for x.
        pred : np.ndarray
            Predictions of the original model for x.
        """

        attack_x, attack_y = self.ai_preprocessor.fit_transform(x, y, pred)
        self._values = self.ai_preprocessor._values  # noqa: SLF001
        self.attack_model.fit(attack_x, attack_y)

    def infer(self, x: np.ndarray, y: np.ndarray, pred: np.ndarray, **kwargs) -> np.ndarray:
        """
        Infer the attacked feature.

        Parameters
        ----------
        x : np.ndarray
            Input to attack. Includes all features except the attacked feature.
        y : np.ndarray
            True labels for x.
        pred : np.ndarray
            Original model's predictions for x.
        values : list
            Possible values for attacked feature. For a single column feature this should be a simple list
            containing all possible values, in increasing order (the smallest value in the 0 index and so
            on). For a multi-column feature (for example 1-hot encoded and then scaled), this should be a
            list of lists, where each internal list represents a column (in increasing order) and the values
            represent the possible values for that column (in increasing order). If not provided, is
            computed from the training data when calling `fit`.

        Returns
        -------
        np.ndarray
            The inferred feature values.
        """

        values: Optional[list] = kwargs.get("values")

        # if provided, override the values computed in fit()
        if values is not None:
            self._values = values

        attack_x = self.ai_preprocessor.transform(x, y, pred)
        predictions = self.attack_model.predict_proba(attack_x).astype(np.float32)

        if self._values is not None:
            if isinstance(self.attack_feature, int):
                predictions = np.array([self._values[np.argmax(arr)] for arr in predictions])
            else:
                i = 0
                for column in predictions.T:
                    for index in range(len(self._values[i])):
                        np.place(column, [column == index], self._values[i][index])
                    i += 1
        return np.array(predictions)

    def _check_params(self) -> None:
        if not isinstance(self.attack_feature, int) and not isinstance(self.attack_feature, slice):
            raise TypeError("Attack feature must be either an integer or a slice object.")

        if isinstance(self.attack_feature, int) and self.attack_feature < 0:
            raise ValueError("Attack feature index must be positive.")

        if self._attack_model_type not in ["nn", "rf"]:
            raise ValueError("Illegal value for parameter `attack_model_type`.")
