"""
This module implements abstract base and mixin classes for estimators in ART.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from holisticai.security.attackers.attribute_inference.mitigation.config import ART_NUMPY_DTYPE


class BaseEstimator(ABC):
    """
    The abstract base class `BaseEstimator` defines the basic requirements of an estimator in ART. The BaseEstimator is
    is the highest abstraction of a machine learning model in ART.

    Parameters
    ----------
    model : object
        The model to be wrapped.
    clip_values : tuple
        Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
    preprocessing_defences : `Preprocessor` or `List[Preprocessor]`
        Preprocessing defence(s) to be applied by the estimator.
    postprocessing_defences : `Postprocessor` or `List[Postprocessor]`
        Postprocessing defence(s) to be applied by the estimator.
    preprocessing : tuple
        Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be used for data preprocessing.
        The first value will be subtracted from the input and the results will be divided by the second value.
    """

    estimator_params = [
        "model",
        "clip_values",
        "preprocessing_defences",
        "postprocessing_defences",
        "preprocessing",
    ]

    def __init__(
        self,
        model,
        clip_values,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0.0, 1.0),
    ):
        self._model = model
        self._clip_values = clip_values

        self.preprocessing = self._set_preprocessing(preprocessing)
        self.preprocessing_defences = self._set_preprocessing_defences(preprocessing_defences)
        self.postprocessing_defences = self._set_postprocessing_defences(postprocessing_defences)
        self.preprocessing_operations = []
        BaseEstimator._update_preprocessing_operations(self)
        BaseEstimator._check_params(self)

    def _update_preprocessing_operations(self):
        from holisticai.security.attackers.attribute_inference.mitigation.defences.preprocessor.preprocessor import (
            Preprocessor,
        )

        self.preprocessing_operations.clear()

        if self.preprocessing_defences is None:
            pass
        elif isinstance(self.preprocessing_defences, Preprocessor):
            self.preprocessing_operations.append(self.preprocessing_defences)
        else:
            self.preprocessing_operations += self.preprocessing_defences

        if self.preprocessing is None:
            pass
        elif isinstance(self.preprocessing, tuple):
            from holisticai.security.attackers.attribute_inference.mitigation.preprocessing.standardisation.numpy import (
                StandardisationMeanStd,
            )

            self.preprocessing_operations.append(
                StandardisationMeanStd(mean=self.preprocessing[0], std=self.preprocessing[1])
            )
        elif isinstance(self.preprocessing, Preprocessor):
            self.preprocessing_operations.append(self.preprocessing)
        else:  # pragma: no cover
            raise ValueError("Preprocessing argument not recognised.")

    @staticmethod
    def _set_preprocessing(preprocessing):
        from holisticai.security.attackers.attribute_inference.mitigation.defences.preprocessor.preprocessor import (
            Preprocessor,
        )

        if preprocessing is None:
            return None
        if isinstance(preprocessing, tuple):
            from holisticai.security.attackers.attribute_inference.mitigation.preprocessing.standardisation.numpy import (
                StandardisationMeanStd,
            )

            return StandardisationMeanStd(mean=preprocessing[0], std=preprocessing[1])  # type: ignore
        if isinstance(preprocessing, Preprocessor):
            return preprocessing

        raise ValueError("Preprocessing argument not recognised.")  # pragma: no cover

    @staticmethod
    def _set_preprocessing_defences(preprocessing_defences):
        from holisticai.security.attackers.attribute_inference.mitigation.defences.preprocessor.preprocessor import (
            Preprocessor,
        )

        if isinstance(preprocessing_defences, Preprocessor):
            return [preprocessing_defences]

        return preprocessing_defences

    @staticmethod
    def _set_postprocessing_defences(postprocessing_defences):
        from holisticai.security.attackers.attribute_inference.mitigation.defences.postprocessor.postprocessor import (
            Postprocessor,
        )

        if isinstance(postprocessing_defences, Postprocessor):
            return [postprocessing_defences]

        return postprocessing_defences

    def set_params(self, **kwargs) -> None:
        """
        Take a dictionary of parameters and apply checks before setting them as attributes.

        Parameters
        ----------
        **kwargs
            A dictionary of attributes.
        """
        for key, value in kwargs.items():
            if key in self.estimator_params:
                if hasattr(type(self), key) and isinstance(getattr(type(self), key), property):
                    if getattr(type(self), key).fset is not None:
                        setattr(self, key, value)
                    else:
                        setattr(self, "_" + key, value)
                elif hasattr(self, "_" + key):
                    setattr(self, "_" + key, value)
                elif key == "preprocessing":
                    setattr(self, key, self._set_preprocessing(value))
                elif key == "preprocessing_defences":
                    setattr(self, key, self._set_preprocessing_defences(value))
                elif key == "postprocessing_defences":
                    setattr(self, key, self._set_postprocessing_defences(value))
                else:
                    setattr(self, key, value)
            else:  # pragma: no cover
                raise ValueError(f"Unexpected parameter `{key}` found in kwargs.")
        self._update_preprocessing_operations()
        self._check_params()

    def get_params(self) -> dict[str, Any]:
        """
        Get all parameters and their values of this estimator.

        Returns
        -------
        dict
            A dictionary of string parameter names to their value.
        """
        params = {}
        for key in self.estimator_params:
            params[key] = getattr(self, key)
        return params

    def _check_params(self) -> None:
        from holisticai.security.attackers.attribute_inference.mitigation.defences.postprocessor.postprocessor import (
            Postprocessor,
        )
        from holisticai.security.attackers.attribute_inference.mitigation.defences.preprocessor.preprocessor import (
            Preprocessor,
        )

        if self._clip_values is not None:
            if len(self._clip_values) != 2:  # pragma: no cover
                raise ValueError(
                    "`clip_values` should be a tuple of 2 floats or arrays containing the allowed data range."
                )
            if np.array(self._clip_values[0] >= self._clip_values[1]).any():  # pragma: no cover
                raise ValueError("Invalid `clip_values`: min >= max.")

            if isinstance(self._clip_values, np.ndarray):
                self._clip_values = self._clip_values.astype(ART_NUMPY_DTYPE)
            else:
                self._clip_values = np.array(self._clip_values, dtype=ART_NUMPY_DTYPE)  # type: ignore

        if isinstance(self.preprocessing_operations, list):
            for preprocess in self.preprocessing_operations:
                if not isinstance(preprocess, Preprocessor):  # pragma: no cover
                    raise TypeError(
                        "All preprocessing defences have to be instance of "
                        "holisticai.security.attackers.attribute_inference.mitigation.defences.preprocessor.preprocessor.Preprocessor."
                    )
        else:  # pragma: no cover
            raise TypeError(
                "All preprocessing defences have to be instance of "
                "holisticai.security.attackers.attribute_inference.mitigation.defences.preprocessor.preprocessor.Preprocessor."
            )

        if isinstance(self.postprocessing_defences, list):
            for postproc_defence in self.postprocessing_defences:
                if not isinstance(postproc_defence, Postprocessor):  # pragma: no cover
                    raise TypeError(
                        "All postprocessing defences have to be instance of "
                        "art.defences.postprocessor.postprocessor.Postprocessor."
                    )
        elif self.postprocessing_defences is None:
            pass
        else:  # pragma: no cover
            raise ValueError(
                "All postprocessing defences have to be instance of "
                "art.defences.postprocessor.postprocessor.Postprocessor."
            )

    @abstractmethod
    def predict(self, x, **kwargs) -> Any:  # lgtm [py/inheritance/incorrect-overridden-signature]
        """
        Perform prediction of the estimator for input `x`.

        Parameters
        ----------
        x : array-like
            Input samples.

        Returns
        -------
        array-like
            Predictions by the model.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, x, y, **kwargs) -> None:  # lgtm [py/inheritance/incorrect-overridden-signature]
        """
        Fit the estimator using the training data `(x, y)`.

        Parameters
        ----------
        x : array-like
            Training data.
        y : array-like
            Target values.
        """
        raise NotImplementedError

    @property
    def model(self):
        """
        Return the model.

        Returns
        -------
        model
            The model.
        """
        return self._model

    @property
    @abstractmethod
    def input_shape(self) -> tuple[int, ...]:
        """
        Return the shape of one input sample.

        Returns
        -------
        tuple
            Shape of one input sample.
        """
        raise NotImplementedError

    @property
    def clip_values(self):
        """
        Return the clip values of the input samples.

        Returns
        -------
        tuple
            Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
        """
        return self._clip_values

    def _apply_preprocessing(self, x, y, fit: bool) -> tuple[Any, Any]:
        """
        Apply all defences and preprocessing operations on the inputs `x` and `y`. This function has to be applied to
        all raw inputs `x` and `y` provided to the estimator.

        Parameters
        ----------
        x : array-like
            Input samples.
        y : array-like
            Target values.
        fit : bool
            `True` if the defences are applied during training.

        Returns
        -------
        tuple
            Tuple of `x` and `y` after applying the defences and standardisation.
        """
        if self.preprocessing_operations:
            for preprocess in self.preprocessing_operations:
                if fit:
                    if preprocess.apply_fit:
                        x, y = preprocess(x, y)
                elif preprocess.apply_predict:
                    x, y = preprocess(x, y)

        return x, y

    def _apply_postprocessing(self, preds, fit: bool) -> np.ndarray:
        """
        Apply all postprocessing defences on model predictions.

        Parameters
        ----------
        preds : array-like
            Model output to be post-processed.
        fit : bool
            `True` if the defences are applied during training.

        Returns
        -------
        array-like
            Post-processed model predictions.
        """
        post_preds = preds.copy()
        if self.postprocessing_defences is not None:
            for defence in self.postprocessing_defences:
                if fit:
                    if defence.apply_fit:
                        post_preds = defence(post_preds)
                elif defence.apply_predict:
                    post_preds = defence(post_preds)

        return post_preds

    def compute_loss(self, x: np.ndarray, y: Any, **kwargs) -> np.ndarray:
        """
        Compute the loss of the estimator for samples `x`.

        Parameters
        ----------
        x : array-like
            Input samples.
        y : array-like
            Target values.

        Returns
        -------
        array-like
            Loss values.
        """
        raise NotImplementedError

    def compute_loss_from_predictions(self, pred: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the loss of the estimator for predictions `pred`.

        Parameters
        ----------
        pred : array-like
            Model predictions.
        y : array-like
            Target values.

        Returns
        -------
        array-like
            Loss values.
        """
        raise NotImplementedError

    def __repr__(self):
        class_name = self.__class__.__name__
        attributes = {}
        for k, value in self.__dict__.items():
            k = k[1:] if k[0] == "_" else k  # noqa: PLW2901
            attributes[k] = value
        attributes = [f"{k}={v}" for k, v in attributes.items()]
        repr_string = class_name + "(" + ", ".join(attributes) + ")"
        return repr_string


class LossGradientsMixin(ABC):
    """
    Mixin abstract base class defining additional functionality for estimators providing loss gradients. An estimator
    of this type can be combined with white-box attacks. This mixin abstract base class has to be mixed in with
    class `BaseEstimator`.
    """

    @abstractmethod
    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        Parameters
        ----------
        x : array-like
            Input samples.
        y : array-like
            Target values.

        Returns
        -------
        array-like
            Loss gradients w.r.t. `x`.
        """
        raise NotImplementedError

    def _apply_preprocessing_gradient(self, x, gradients, fit=False):
        """
        Apply the backward pass to the gradients through all normalization and preprocessing defences that have been
        applied to `x` and `y` in the forward pass. This function has to be applied to all gradients w.r.t. `x`
        calculated by the estimator.

        Parameters
        ----------
        x : array-like
            Input samples.
        gradients : array-like
            Gradients w.r.t. `x`.
        fit : bool
            `True` if the defences are applied during training.

        Returns
        -------
        array-like
            Gradients after backward pass through normalization and preprocessing defences.
        """
        if self.preprocessing_operations:
            for preprocess in self.preprocessing_operations[::-1]:
                if fit:
                    if preprocess.apply_fit:
                        gradients = preprocess.estimate_gradient(x, gradients)
                elif preprocess.apply_predict:
                    gradients = preprocess.estimate_gradient(x, gradients)

        return gradients


class DecisionTreeMixin(ABC):
    """
    Mixin abstract base class defining additional functionality for decision-tree-based estimators. This mixin abstract
    base class has to be mixed in with class `BaseEstimator`.
    """

    @abstractmethod
    def get_trees(self):
        """
        Get the decision trees.

        Returns
        -------
        list
            A list of decision trees.
        """
        raise NotImplementedError
