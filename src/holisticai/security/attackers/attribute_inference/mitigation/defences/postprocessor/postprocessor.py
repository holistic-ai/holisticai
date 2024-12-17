# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements the abstract base class for defences that post-process classifier output.
"""

from __future__ import annotations

import abc

import numpy as np


class Postprocessor(abc.ABC):
    """
    Abstract base class for postprocessing defences. Postprocessing defences are not included in the loss function
    evaluation for loss gradients or the calculation of class gradients.

    Parameters
    ----------
    is_fitted : bool
        Whether the postprocessor has already been fitted.
    apply_fit : bool
        Whether the postprocessor should be applied at training time.
    apply_predict : bool
        Whether the postprocessor should be applied at test time.
    """

    params: list[str] = []

    def __init__(self, is_fitted: bool = False, apply_fit: bool = True, apply_predict: bool = True) -> None:
        self._is_fitted = bool(is_fitted)
        self._apply_fit = bool(apply_fit)
        self._apply_predict = bool(apply_predict)
        Postprocessor._check_params(self)

    @property
    def is_fitted(self) -> bool:
        """
        Return the state of the postprocessing object.

        Returns
        -------
        bool
            `True` if the postprocessing model has been fitted (if this applies).
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
    def __call__(self, preds: np.ndarray) -> np.ndarray:
        """
        Perform model postprocessing and return postprocessed output.

        Parameters
        ----------
        preds : np.ndarray
            Model output to be postprocessed.

        Returns
        -------
        np.ndarray
            Postprocessed model output.
        """
        raise NotImplementedError

    def fit(self, preds: np.ndarray, **kwargs) -> None:
        """
        Fit the parameters of the postprocessor if it has any.

        Parameters
        ----------
        preds : np.ndarray
            Training set to fit the postprocessor.
        kwargs
            Other parameters.
        """

    def set_params(self, **kwargs) -> None:
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

    def _check_params(self) -> None:
        pass
