from __future__ import annotations

import abc
import logging
from typing import Any, Optional, Union

import numpy as np
from holisticai.security.attackers.attribute_inference.exceptions import EstimatorError
from holisticai.security.attackers.attribute_inference.utils import is_estimator_valid

logger = logging.getLogger(__name__)


class InputFilter(abc.ABCMeta):  # pragma: no cover
    """
    Metaclass to ensure that inputs are ndarray for all of the subclass generate and extract calls

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
        def make_replacement(fdict, func_name):
            """
            This function overrides creates replacement functions dynamically.

            Parameters
            ----------
            fdict : dict
                The class dictionary.
            func_name : str
                The name of the function to override.

            Returns
            -------
            replacement_function : function
                The replacement function.
            """

            def replacement_function(self, *args, **kwargs):
                if len(args) > 0:
                    lst = list(args)

                if "x" in kwargs:
                    if not isinstance(kwargs["x"], np.ndarray):
                        kwargs["x"] = np.array(kwargs["x"])
                elif not isinstance(args[0], np.ndarray):
                    lst[0] = np.array(args[0])

                if "y" in kwargs:
                    if kwargs["y"] is not None and not isinstance(kwargs["y"], np.ndarray):
                        kwargs["y"] = np.array(kwargs["y"])
                elif len(args) == 2 and not isinstance(args[1], np.ndarray):
                    lst[1] = np.array(args[1])

                if len(args) > 0:
                    args = tuple(lst)
                return fdict[func_name](self, *args, **kwargs)

            replacement_function.__doc__ = fdict[func_name].__doc__
            replacement_function.__name__ = "new_" + func_name
            return replacement_function

        replacement_list = ["generate", "extract"]
        for item in replacement_list:
            if item in clsdict:
                new_function = make_replacement(clsdict, item)
                setattr(cls, item, new_function)


class Attack(abc.ABC):
    """
    Abstract base class for all attack abstract base classes.

    Parameters
    ----------
    estimator : :class:`.holisticai.wrappers.estimator.BaseEstimator`
        An estimator.
    """

    attack_params: list[str] = []
    # The _estimator_requirements define the requirements an estimator must satisfy to be used as a target for an
    # attack. They should be a tuple of requirements, where each requirement is either a class the estimator must
    # inherit from, or a tuple of classes which define a union, i.e. the estimator must inherit from at least one class
    # in the requirement tuple.
    _estimator_requirements: Optional[Union[tuple[Any, ...], tuple[()]]] = None

    def __init__(self, estimator):
        super().__init__()

        if self.estimator_requirements is None:
            raise ValueError("Estimator requirements have not been defined in `_estimator_requirements`.")

        if not is_estimator_valid(estimator, self._estimator_requirements):
            raise EstimatorError(self.__class__, self.estimator_requirements, estimator)

        self._estimator = estimator

    @property
    def estimator(self):
        """
        The estimator.

        Returns
        -------
        object
            The estimator.
        """
        return self._estimator

    @property
    def estimator_requirements(self):
        """
        The estimator requirements.

        Returns
        -------
        tuple
            The estimator requirements.
        """
        return self._estimator_requirements

    def set_params(self, **kwargs) -> None:
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        Parameters
        ----------
        **kwargs
            A dictionary of attack-specific parameters.
        """
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
        self._check_params()


class InferenceAttack(Attack):
    """
    Abstract base class for inference attack classes.

    Parameters
    ----------
    estimator : object
        A trained estimator targeted for inference attack.
    """

    def __init__(self, estimator):
        super().__init__(estimator)

    @abc.abstractmethod
    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer sensitive attributes from the targeted estimator. This method
        should be overridden by all concrete inference attack implementations.

        Parameters
        ----------
        x : np.ndarray
            An array with reference inputs to be used in the attack.
        y : np.ndarray, optional
            Labels for `x`. This parameter is only used by some of the attacks.

        Returns
        -------
        np.ndarray
            An array holding the inferred attribute values.
        """
        raise NotImplementedError


class AttributeInferenceAttack(InferenceAttack):
    """
    Abstract base class for attribute inference attack classes.

    Parameters
    ----------
    estimator : object
        A trained estimator targeted for inference attack.
    attack_feature : int or slice
        The index of the feature to be attacked.
    """

    attack_params = [*InferenceAttack.attack_params, "attack_feature"]

    def __init__(self, estimator, attack_feature: Union[int, slice] = 0):
        super().__init__(estimator)
        self.attack_feature = attack_feature

    @abc.abstractmethod
    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer sensitive attributes from the targeted estimator. This method
        should be overridden by all concrete inference attack implementations.

        Parameters
        ----------
        x : np.ndarray
            An array with reference inputs to be used in the attack.
        y : np.ndarray, optional
            Labels for `x`. This parameter is only used by some of the attacks.

        Returns
        -------
        np.ndarray
            An array holding the inferred attribute values.
        """
        raise NotImplementedError
