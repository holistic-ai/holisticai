# from typing import Optional, Union, TYPE_CHECKING
import numpy as np
from holisticai.security.attackers.attribute_inference.attack import AttributeInferenceAttack
from holisticai.security.attackers.attribute_inference.dataset_utils import AttributeInferenceDataPreprocessor
from holisticai.security.attackers.attribute_inference.utils import get_attack_model, get_feature_index
from sklearn.base import ClassifierMixin


class AttributeInferenceBaseline(AttributeInferenceAttack):
    """
    Implementation of a baseline attribute inference, not using a model.

    The idea is to train a simple neural network to learn the attacked feature from the rest of the features. Should
    be used to compare with other attribute inference results.

    Parameters
    ----------
    attack_model_type : str
        The type of default attack model to train, optional. Should be one of `nn` (for neural network, default) or `rf`
        (for random forest). If `attack_model` is supplied, this option will be ignored.
    attack_model : object
        The attack model to train, optional. If none is provided, a default model will be created.
    attack_feature : int or slice
        The index of the feature to be attacked or a slice representing multiple indexes in case of a one-hot encoded
        feature.
    """

    _estimator_requirements = ()

    def __init__(
        self,
        attack_model_type="nn",
        attack_model=None,
        attack_feature=0,
    ):
        super().__init__(estimator=None, attack_feature=attack_feature)

        self._values = None
        self._nb_classes = None

        if attack_model:
            if ClassifierMixin not in type(attack_model).__mro__:
                raise ValueError("Attack model must be of type Classifier.")
            self.attack_model = attack_model
        else:
            self.attack_model = get_attack_model(attack_model_type)

        self._check_params()
        self.attack_feature = get_feature_index(self.attack_feature)
        self.ai_preprocessor = AttributeInferenceDataPreprocessor(attack_feature=attack_feature)

    def fit(self, x: np.ndarray) -> None:
        """
        Train the attack model.

        Parameters
        ----------
        x : np.ndarray
            Input to training process. Includes all features used to train the original model.
        """
        # train attack model
        attack_x, attack_y = self.ai_preprocessor.fit_transform(x)
        self._values = self.ai_preprocessor._values  # noqa: SLF001
        self.attack_model.fit(attack_x, attack_y)

    def infer(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Infer the attacked feature.

        Parameters
        ----------
        x : np.ndarray
            Input to attack. Includes all features except the attacked feature.
        values : list
            Possible values for attacked feature. For a single column feature this should be a simple list containing
            all possible values, in increasing order (the smallest value in the 0 index and so on). For a multi-column
            feature (for example 1-hot encoded and then scaled), this should be a list of lists, where each internal
            list represents a column (in increasing order) and the values represent the possible values for that column
            (in increasing order).

        Returns
        -------
        np.ndarray
            The inferred feature values.
        """
        # if values are provided, override the values computed in fit()
        values = kwargs.get("values", self._values)
        attack_x = self.ai_preprocessor.transform(x)
        predictions = self.attack_model.predict_proba(attack_x).astype(np.float32)

        if values is not None:
            if isinstance(self.attack_feature, int):
                predictions = np.array([values[np.argmax(arr)] for arr in predictions])
            else:
                for value, column in zip(values, predictions.T):
                    for index in range(len(value)):
                        np.place(column, [column == index], value[index])
        return np.array(predictions)

    def _check_params(self) -> None:
        if not isinstance(self.attack_feature, int) and not isinstance(self.attack_feature, slice):
            raise TypeError("Attack feature must be either an integer or a slice object.")

        if isinstance(self.attack_feature, int) and self.attack_feature < 0:
            raise ValueError("Attack feature index must be non-negative.")
