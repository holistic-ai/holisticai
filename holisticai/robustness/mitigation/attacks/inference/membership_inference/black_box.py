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
This module implements membership inference attacks.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from holisticai.robustness.mitigation.attacks.attack import MembershipInferenceAttack
from holisticai.robustness.mitigation.utils.formatting import (
    check_and_transform_label_format,
)
from holisticai.wrappers.classification.classifier import ClassifierMixin
from holisticai.wrappers.estimator import BaseEstimator
from holisticai.wrappers.regression import RegressorMixin

if TYPE_CHECKING:
    from holisticai.wrappers.formatting import CLASSIFIER_TYPE, REGRESSOR_TYPE

logger = logging.getLogger(__name__)


class MembershipInferenceBlackBox(MembershipInferenceAttack):
    """
    Implementation of a learned black-box membership inference attack.

    This implementation can use as input to the learning process probabilities/logits or losses,
    depending on the type of model and provided configuration.
    """

    attack_params = MembershipInferenceAttack.attack_params + [
        "input_type",
        "attack_model_type",
        "attack_model",
    ]
    _estimator_requirements = (BaseEstimator, (ClassifierMixin, RegressorMixin))

    def __init__(
        self,
        estimator: Union["CLASSIFIER_TYPE", "REGRESSOR_TYPE"],
        input_type: str = "prediction",
        attack_model_type: str = "nn",
        attack_model: Optional[Any] = None,
    ):
        """
        Create a MembershipInferenceBlackBox attack instance.

        :param estimator: Target estimator.
        :param attack_model_type: the type of default attack model to train, optional. Should be one of `nn` (for neural
                                  network, default), `rf` (for random forest) or `gb` (gradient boosting). If
                                  `attack_model` is supplied, this option will be ignored.
        :param input_type: the type of input to train the attack on. Can be one of: 'prediction' or 'loss'. Default is
                           `prediction`. Predictions can be either probabilities or logits, depending on the return type
                           of the model. If the model is a regressor, only `loss` can be used.
        :param attack_model: The attack model to train, optional. If none is provided, a default model will be created.
        """

        super().__init__(estimator=estimator)
        self.input_type = input_type
        self.attack_model_type = attack_model_type
        self.attack_model = attack_model

        self._regressor_model = RegressorMixin in type(self.estimator).__mro__

        self._check_params()

        if self.attack_model:
            self.default_model = False
            self.attack_model_type = "None"
        else:
            self.default_model = True
            if self.attack_model_type == "nn":
                from .nn_atack_model import MembershipInferenceAttackModel, Trainer

                if self._regressor_model:
                    self.attack_model = MembershipInferenceAttackModel(
                        1, num_features=1
                    )
                else:
                    num_classes = estimator.nb_classes  # type: ignore
                    if self.input_type == "prediction":
                        self.attack_model = MembershipInferenceAttackModel(num_classes)
                    else:
                        self.attack_model = MembershipInferenceAttackModel(
                            num_classes, num_features=1
                        )

                self.attack_trainer = Trainer(
                    learning_rate=0.0001,
                    batch_size=100,
                    epochs=100,
                    attack_model=self.attack_model,
                )

            elif self.attack_model_type == "rf":
                self.attack_model = RandomForestClassifier()

            elif self.attack_model_type == "gb":
                self.attack_model = GradientBoostingClassifier()

    def fit(  # pylint: disable=W0613
        self,
        x: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        pred: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Train the attack model.

        :param x: Records that were used in training the target estimator. Can be None if supplying `pred`.
        :param y: True labels for `x`.
        :param pred: Estimator predictions for the records, if not supplied will be generated by calling the estimators'
                     `predict` function. Only relevant for input_type='prediction'.
        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member.
        """

        # Create attack dataset
        # uses final probabilities/logits
        if self.input_type == "prediction":
            if pred is None:
                features = self.estimator.predict_proba(x).astype(np.float32)
            else:
                features = pred.astype(np.float32)

        # only for models with loss
        elif self.input_type == "loss":
            if x is not None:
                features = (
                    self.estimator.compute_loss(x, y).astype(np.float32).reshape(-1, 1)
                )
            else:
                features = (
                    self.estimator.compute_loss_from_predictions(pred, y)
                    .astype(np.float32)
                    .reshape(-1, 1)
                )

        else:  # pragma: no cover
            raise ValueError("Illegal value for parameter `input_type`.")

        if self._regressor_model:
            x_2 = x_2.astype(np.float32).reshape(-1, 1)

        if not self._regressor_model:
            y = check_and_transform_label_format(
                y, nb_classes=self.estimator.nb_classes, return_one_hot=True
            )

        x_1 = features
        x_2 = y
        if self.default_model and self.attack_model_type == "nn":
            from .nn_atack_model import AttackDataset

            dataset = AttackDataset(x_1=x_1, x_2=x_2, y=m)
            self.attack_trainer.train(dataset=dataset)
        else:
            y_ready = check_and_transform_label_format(
                m, nb_classes=2, return_one_hot=False
            )
            self.attack_model.fit(np.c_[x_1, x_2], y_ready.ravel())  # type: ignore

    def infer(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """
        Infer membership in the training set of the target estimator.

        :param x: Input records to attack. Can be None if supplying `pred`.
        :param y: True labels for `x`.
        :param pred: Estimator predictions for the records, if not supplied will be generated by calling the estimators'
                     `predict` function. Only relevant for input_type='prediction'.
        :param probabilities: a boolean indicating whether to return the predicted probabilities per class, or just
                              the predicted class.
        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member,
                 or class probabilities.
        """
        if "pred" in kwargs:
            pred = kwargs.get("pred")
        else:
            pred = None

        if "probabilities" in kwargs:
            probabilities = kwargs.get("probabilities")
        else:
            probabilities = False

        if y is None:  # pragma: no cover
            raise ValueError("MembershipInferenceBlackBox requires true labels `y`.")
        if x is None and pred is None:
            raise ValueError("Must supply either x or pred")

        if self.estimator.input_shape is not None and x is not None:  # pragma: no cover
            if self.estimator.input_shape[0] != x.shape[1]:
                raise ValueError("Shape of x does not match input_shape of estimator")

        if not self._regressor_model:
            y = check_and_transform_label_format(
                y, nb_classes=self.estimator.nb_classes, return_one_hot=True
            )

        if y is None:
            raise ValueError("None value detected.")

        if x is not None and y.shape[0] != x.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in x and y do not match")
        if pred is not None and y.shape[0] != pred.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in pred and y do not match")

        if self.input_type == "prediction":
            if pred is None:
                features = self.estimator.predict_proba(x).astype(np.float32)
            else:
                features = pred.astype(np.float32)
        elif self.input_type == "loss":
            if x is not None:
                features = (
                    self.estimator.compute_loss(x, y).astype(np.float32).reshape(-1, 1)
                )
            else:
                try:
                    features = (
                        self.estimator.compute_loss_from_predictions(pred, y)
                        .astype(np.float32)
                        .reshape(-1, 1)
                    )
                except NotImplementedError as err:
                    raise ValueError(
                        "For loss input type and no x, the estimator must implement 'compute_loss_from_predictions' "
                        "method"
                    ) from err

        if self._regressor_model:
            y = y.astype(np.float32).reshape(-1, 1)

        if self.default_model and self.attack_model_type == "nn":
            from .nn_atack_model import AttackDataset

            dataset = AttackDataset(x_1=features, x_2=y)
            inferred = self.attack_trainer.predict(dataset)

        elif not self.default_model:
            inferred = self.attack_model.predict_proba(np.c_[features, y])[:, [1]]

        else:
            inferred = self.attack_model.predict_proba(np.c_[features, y])[:, [1]]

        if probabilities:
            inferred_return = inferred
        else:
            inferred_return = np.round(inferred)

        return inferred_return

    def _check_params(self) -> None:
        if self.input_type not in ["prediction", "loss"]:
            raise ValueError("Illegal value for parameter `input_type`.")

        if self._regressor_model:
            if self.input_type != "loss":
                raise ValueError(
                    "Illegal value for parameter `input_type` when estimator is a regressor."
                )

        if self.attack_model_type not in ["nn", "rf", "gb"]:
            raise ValueError("Illegal value for parameter `attack_model_type`.")

        if self.attack_model:
            if ClassifierMixin not in type(self.attack_model).__mro__:
                raise TypeError("Attack model must be of type Classifier.")
