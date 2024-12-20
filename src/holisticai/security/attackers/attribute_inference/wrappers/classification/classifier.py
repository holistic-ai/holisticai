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
This module implements mixin abstract base classes defining properties for all classifiers in ART.
"""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import Optional, Union

import numpy as np
from holisticai.security.attackers.attribute_inference.wrappers.estimator import (
    BaseEstimator,
    DecisionTreeMixin,
    LossGradientsMixin,
)


class InputFilter(ABCMeta):
    """
    Metaclass to ensure that inputs are ndarray for all of the subclass generate and extract calls.

    This metaclass is used to ensure that the input to all generate and extract methods is an ndarray. This is
    necessary because the input to these methods is often a list or a pandas DataFrame, and the methods themselves
    are not always implemented to handle these types of inputs. This metaclass overrides the generate and extract
    methods with new methods that ensure the input is an ndarray.

    Parameters
    ----------
    name : str
        The name of the class.
    bases : tuple
        The base classes of the class.
    clsdict : dict
        The class dictionary.
    """

    def __init__(cls, name, bases, clsdict):  # noqa: ARG003
        def make_replacement(fdict, func_name, has_y):
            """
            This function overrides creates replacement functions dynamically.

            Parameters
            ----------
            fdict : dict
                The class dictionary.
            func_name : str
                The name of the function to override.
            has_y : bool
                Whether the function has a y parameter.

            Returns
            -------
            replacement_function : function
                The replacement function.
            """

            def replacement_function(self, *args, **kwargs):
                """
                Replacement function.

                Parameters
                ----------
                self : object
                    The object.
                args : list
                    The arguments.
                kwargs : dict
                    The keyword arguments.

                Returns
                -------
                result : object
                    The result.
                """
                if len(args) > 0:
                    lst = list(args)

                if "X" in kwargs:
                    if not isinstance(kwargs["X"], np.ndarray):
                        np.array(kwargs["X"])
                    kwargs["x"] = kwargs["X"].copy()
                    del kwargs["X"]

                elif "x" in kwargs:  # pragma: no cover
                    if not isinstance(kwargs["x"], np.ndarray):
                        kwargs["x"] = np.array(kwargs["x"])
                elif not isinstance(args[0], np.ndarray):
                    lst[0] = np.array(args[0])

                if "y" in kwargs:  # pragma: no cover
                    if kwargs["y"] is not None and not isinstance(kwargs["y"], np.ndarray):
                        kwargs["y"] = np.array(kwargs["y"])
                elif has_y:  # pragma: no cover  # noqa: SIM102
                    if not isinstance(args[1], np.ndarray):
                        lst[1] = np.array(args[1])

                if len(args) > 0:
                    args = tuple(lst)
                return fdict[func_name](self, *args, **kwargs)

            replacement_function.__doc__ = fdict[func_name].__doc__
            replacement_function.__name__ = "new_" + func_name
            return replacement_function

        replacement_list_no_y = ["predict"]
        replacement_list_has_y = ["fit"]

        for item in replacement_list_no_y:
            if item in clsdict:
                new_function = make_replacement(clsdict, item, False)
                setattr(cls, item, new_function)
        for item in replacement_list_has_y:
            if item in clsdict:
                new_function = make_replacement(clsdict, item, True)
                setattr(cls, item, new_function)


class ClassifierMixin(ABC, metaclass=InputFilter):
    """
    Mixin abstract base class defining functionality for classifiers.

    This abstract base class defines the properties of a classifier and provides functionality to set and get the number
    of classes in the data. It also provides an interface to clone the classifier for refitting.

    Parameters
    ----------
    nb_classes : int
        The number of output classes.
    """

    estimator_params = ["nb_classes"]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)  # type: ignore
        self._nb_classes: int = -1

    @property
    def nb_classes(self) -> int:
        """
        Return the number of output classes.

        Returns
        -------
        nb_classes : int
            Number of classes in the data.
        """
        return self._nb_classes  # type: ignore

    @nb_classes.setter
    def nb_classes(self, nb_classes: int):
        """
        Set the number of output classes.

        Parameters
        ----------
        nb_classes : int
            Number of classes in the data.

        Raises
        ------
        ValueError
            If `nb_classes` is less than 2.
        """
        if nb_classes is None or nb_classes < 2:
            raise ValueError("nb_classes must be greater than or equal to 2.")

        self._nb_classes = nb_classes

    def clone_for_refitting(self):
        """
        Clone classifier for refitting.
        """
        raise NotImplementedError


class ClassGradientsMixin(ABC):
    """
    Mixin abstract base class defining classifiers providing access to class gradients. A classifier of this type can
    be combined with certain white-box attacks. This mixin abstract base class has to be mixed in with
    class `Classifier`.
    """

    @abstractmethod
    def class_gradient(self, x: np.ndarray, label: Optional[Union[int, list[int]]] = None, **kwargs) -> np.ndarray:
        """
        Compute per-class derivatives w.r.t. `x`.

        Parameters
        ----------
        x : np.ndarray
            Samples.
        label : int, list of int, or None
            Index of a specific per-class derivative. If an integer is provided, the gradient of that class output is
            computed for all samples. If multiple values as provided, the first dimension should match the batch size of
            `x`, and each value will be used as target for its corresponding sample in `x`. If `None`, then gradients
            for all classes will be computed for each sample.

        Returns
        -------
        np.ndarray
            Gradients of input features w.r.t. each class in the form `(batch_size, nb_classes, input_shape)` when
            computing for all classes, otherwise shape becomes `(batch_size, 1, input_shape)` when `label` parameter is
            specified.
        """
        raise NotImplementedError


class Classifier(ClassifierMixin, BaseEstimator, ABC):
    """
    Typing variable definition.
    """

    estimator_params = BaseEstimator.estimator_params + ClassifierMixin.estimator_params


class ClassifierLossGradients(ClassifierMixin, LossGradientsMixin, BaseEstimator, ABC):
    """
    Typing variable definition.
    """

    estimator_params = BaseEstimator.estimator_params + ClassifierMixin.estimator_params


class ClassifierClassLossGradients(ClassGradientsMixin, ClassifierMixin, LossGradientsMixin, BaseEstimator, ABC):
    """
    Typing variable definition.
    """

    estimator_params = BaseEstimator.estimator_params + ClassifierMixin.estimator_params


class ClassifierDecisionTree(DecisionTreeMixin, ClassifierMixin, BaseEstimator, ABC):
    """
    Typing variable definition.
    """

    estimator_params = BaseEstimator.estimator_params + ClassifierMixin.estimator_params
