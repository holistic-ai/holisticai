# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the abstract base class for defences that pre-process input data.
"""

from __future__ import annotations

import abc
from typing import Any, Optional

import numpy as np


class Preprocessor(abc.ABC):
    """
    Abstract base class for preprocessing defences.

    By default, the gradient is estimated using BPDA with the identity function.
        To modify, override `estimate_gradient`

    Parameters
    ----------
    is_fitted : bool
        Whether the preprocessor has already been fitted.
    apply_fit : bool
        Whether the preprocessor should be applied at training time.
    apply_predict : bool
        Whether the preprocessor should be applied at test time.
    """

    params: list[str] = []

    def __init__(self, is_fitted: bool = False, apply_fit: bool = True, apply_predict: bool = True) -> None:
        self._is_fitted = bool(is_fitted)
        self._apply_fit = bool(apply_fit)
        self._apply_predict = bool(apply_predict)

    @property
    def is_fitted(self) -> bool:
        """
        Return the state of the preprocessing object.

        Returns
        -------
        bool
            `True` if the preprocessing model has been fitted (if this applies).
        """
        return self._is_fitted

    @property
    def apply_fit(self) -> bool:
        """
        Property of the defence indicating if it should be applied at training time.

        Returns
        -------
        bool
            `True` if the defence should be applied when fitting a model, `False` otherwise.
        """
        return self._apply_fit

    @property
    def apply_predict(self) -> bool:
        """
        Property of the defence indicating if it should be applied at test time.

        Returns
        -------
        bool
            `True` if the defence should be applied at prediction time, `False` otherwise.
        """
        return self._apply_predict

    @abc.abstractmethod
    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Perform data preprocessing and return preprocessed data as tuple.

        Parameters
        ----------
        x : np.ndarray
            Dataset to be preprocessed.
        y : np.ndarray
            Labels to be preprocessed.

        Returns
        -------
        np.ndarray
            Preprocessed data.
        np.ndarray
            Preprocessed labels.
        """
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Fit the parameters of the data preprocessor if it has any.

        Parameters
        ----------
        x : np.ndarray
            Training set to fit the preprocessor.
        y : np.ndarray
            Labels for the training set.
        kwargs
            Other parameters.
        """

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:  # noqa: ARG002
        """
        Provide an estimate of the gradients of the defence for the backward pass. If the defence is not differentiable,
        this is an estimate of the gradient, most often replacing the computation performed by the defence with the
        identity function (the default).

        Parameters
        ----------
        x : np.ndarray
            Input data for which the gradient is estimated. First dimension is the batch size.
        grad : np.ndarray
            Gradient value so far.

        Returns
        -------
        np.ndarray
            The gradient (estimate) of the defence.
        """
        return grad

    def set_params(self, **kwargs) -> None:  # pragma: no cover
        """
        Take in a dictionary of parameters and apply checks before saving them as attributes.

        Parameters
        ----------
        kwargs
            Dictionary of parameters to apply checks to.
        """
        for key, value in kwargs.items():
            if key in self.params:
                setattr(self, key, value)
        self._check_params()

    def _check_params(self) -> None:  # pragma: no cover
        pass

    def forward(self, x: Any, y: Any = None) -> tuple[Any, Any]:
        """
        Perform data preprocessing and return preprocessed data.

        Parameters
        ----------
        x : Any
            Dataset to be preprocessed.
        y : Any
            Labels to be preprocessed.

        Returns
        -------
        Any
            Preprocessed data.
        """
        raise NotImplementedError
