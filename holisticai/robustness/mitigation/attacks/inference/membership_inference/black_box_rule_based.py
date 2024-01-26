"""
This module implements membership inference attacks.
"""
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from holisticai.robustness.mitigation.attacks.attack import MembershipInferenceAttack
from holisticai.robustness.mitigation.utils.formatting import (
    check_and_transform_label_format,
)
from holisticai.wrappers.classification.classifier import ClassifierMixin
from holisticai.wrappers.estimator import BaseEstimator

if TYPE_CHECKING:
    from holisticai.wrappers.formatting import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class MembershipInferenceBlackBoxRuleBased(MembershipInferenceAttack):
    """
    Implementation of a simple, rule-based black-box membership inference attack.

    This implementation uses the simple rule: if the model's prediction for a sample is correct, then it is a
    member. Otherwise, it is not a member.
    """

    attack_params = MembershipInferenceAttack.attack_params
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self, classifier: "CLASSIFIER_TYPE"):
        """
        Create a MembershipInferenceBlackBoxRuleBased attack instance.

        :param classifier: Target classifier.
        """
        super().__init__(estimator=classifier)

    def infer(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """
        Infer membership in the training set of the target estimator.

        :param x: Input records to attack.
        :param y: True labels for `x`.
        :param probabilities: a boolean indicating whether to return the predicted probabilities per class, or just
                              the predicted class.
        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member,
                 or class probabilities.
        """
        if y is None:  # pragma: no cover
            raise ValueError(
                "MembershipInferenceBlackBoxRuleBased requires true labels `y`."
            )

        if self.estimator.input_shape is not None:  # pragma: no cover
            if self.estimator.input_shape[0] != x.shape[1]:
                raise ValueError("Shape of x does not match input_shape of classifier")

        if "probabilities" in kwargs:
            probabilities = kwargs.get("probabilities")
        else:
            probabilities = False

        y = check_and_transform_label_format(
            y, nb_classes=len(np.unique(y)), return_one_hot=True
        )
        if y is None:
            raise ValueError("None value detected.")
        if y.shape[0] != x.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in x and y do not match")

        # get model's predictions for x
        y_pred = self.estimator.predict_proba(x=x)
        predicted_class = (np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)).astype(
            int
        )
        if probabilities:
            # use y_pred as the probability if binary classification, otherwise just use 1
            if y_pred.shape[1] == 2:
                pred_prob = np.max(y_pred, axis=1)
                prob = np.zeros((predicted_class.shape[0], 2))
                prob[:, predicted_class] = pred_prob
                prob[:, np.ones_like(predicted_class) - predicted_class] = (
                    np.ones_like(pred_prob) - pred_prob
                )
            else:
                # simply returns probability 1 for the predicted class and 0 for the other class
                prob_none = check_and_transform_label_format(
                    predicted_class, nb_classes=2, return_one_hot=True
                )
                if prob_none is not None:
                    prob = prob_none
            return prob
        return predicted_class
