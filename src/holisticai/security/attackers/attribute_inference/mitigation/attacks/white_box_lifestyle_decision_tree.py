"""
This module implements attribute inference attacks.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from holisticai.security.attackers.attribute_inference.attack import AttributeInferenceAttack
from holisticai.security.attackers.attribute_inference.wrappers.classification.scikitlearn import (
    ScikitlearnDecisionTreeClassifier,
)
from holisticai.security.attackers.attribute_inference.wrappers.regression.scikitlearn import (
    ScikitlearnDecisionTreeRegressor,
)

logger = logging.getLogger(__name__)


class AttributeInferenceWhiteBoxLifestyleDecisionTree(AttributeInferenceAttack):
    """
    Implementation of Fredrikson et al. white box inference attack for decision trees.

    Assumes that the attacked feature is discrete or categorical, with limited number of possible values. For example:
    a boolean feature.

    Parameters
    ----------
    estimator : object
        Target estimator.
    attack_feature : int
        The index of the feature to be attacked.

    References
    ----------
    .. [1] Fredrikson, M., Jha, S., & Ristenpart, T. (2015, August). Model inversion attacks that exploit confidence\\
          information and basic countermeasures. In Proceedings of the 22nd ACM SIGSAC Conference on Computer and\\
              Communications Security (pp. 1322-1333).

    | Paper link: https://dl.acm.org/doi/10.1145/2810103.2813677
    """

    _estimator_requirements = ((ScikitlearnDecisionTreeClassifier, ScikitlearnDecisionTreeRegressor),)

    def __init__(self, estimator, attack_feature: int = 0):
        super().__init__(estimator=estimator, attack_feature=attack_feature)
        self.attack_feature: int
        self._check_params()

    def infer(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Infer the attacked feature.

        Parameters
        ----------
        x : np.ndarray
            Input to attack. Includes all features except the attacked feature.
        kwargs : dict
            Possible values for attacked feature and prior distributions of attacked feature values.

        Returns
        -------
        np.ndarray
            The inferred feature values.
        """
        priors: Optional[list] = kwargs.get("priors")
        values: Optional[list] = kwargs.get("values")

        # Checks:
        if self.estimator.input_shape[0] != x.shape[1] + 1:  # pragma: no cover
            raise ValueError("Number of features in x + 1 does not match input_shape of classifier")
        if priors is None or values is None:  # pragma: no cover
            raise ValueError("`priors` and `values` are required as inputs.")
        if len(priors) != len(values):  # pragma: no cover
            raise ValueError("Number of priors does not match number of values")

        n_samples = x.shape[0]

        # Calculate phi for each possible value of the attacked feature
        # phi is the total number of samples in all tree leaves corresponding to this value
        phi = self._calculate_phi(x, values, n_samples)

        # Will contain the probability of each value
        prob_values = []

        for i, value in enumerate(values):
            # prepare data with the given value in the attacked feature
            v_full = np.full((n_samples, 1), value).astype(x.dtype)
            x_value = np.concatenate((x[:, : self.attack_feature], v_full), axis=1)
            x_value = np.concatenate((x_value, x[:, self.attack_feature :]), axis=1)

            # find the relative probability of this value for all samples being attacked
            prob_value = [
                (
                    (self.estimator.get_samples_at_node(self.estimator.get_decision_path([row])[-1]) / n_samples)
                    * priors[i]
                    / phi[i]
                )
                for row in x_value
            ]
            prob_values.append(prob_value)

        # Choose the value with highest probability for each sample
        return np.array([values[np.argmax(list(prob))] for prob in zip(*prob_values)])

    def _calculate_phi(self, x, values, n_samples):
        phi = []
        for value in values:
            v_full = np.full((n_samples, 1), value).astype(x.dtype)
            x_value = np.concatenate((x[:, : self.attack_feature], v_full), axis=1)
            x_value = np.concatenate((x_value, x[:, self.attack_feature :]), axis=1)
            nodes_value = {}

            for row in x_value:
                # get leaf ids (no duplicates)
                node_id = self.estimator.get_decision_path([row])[-1]
                nodes_value[node_id] = self.estimator.get_samples_at_node(node_id)
            # sum sample numbers
            num_value = sum(nodes_value.values()) / n_samples
            phi.append(num_value)

        return phi

    def _check_params(self) -> None:
        if self.attack_feature < 0:
            raise ValueError("Attack feature must be positive.")
