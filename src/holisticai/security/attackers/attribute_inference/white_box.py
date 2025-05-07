"""
This module implements attribute inference attacks.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


class AttributeInferenceWhiteBoxDecisionTree():
    """
    A variation of the method proposed by of Fredrikson et al. in:
    https://dl.acm.org/doi/10.1145/2810103.2813677

    Assumes the availability of the attacked model's predictions for the samples under attack, in addition to access to
    the model itself and the rest of the feature values. If this is not available, the true class label of the samples
    may be used as a proxy. Also assumes that the attacked feature is discrete or categorical, with limited number of
    possible values. For example: a boolean feature.

    Parameters
    ----------
    classifier : ScikitlearnDecisionTreeClassifier
        Target classifier.
    attack_feature : int
        The index of the feature to be attacked.

    References
    ----------
    .. [1] Fredrikson, M., Jha, S., & Ristenpart, T. (2015, August). Model inversion attacks that exploit confidence
           information and basic countermeasures. In Proceedings of the 22nd ACM SIGSAC Conference on Computer and
           Communications Security (pp. 1322-1333).
    """
    def __init__(self, classifier, attack_feature: int = 0):
        self.attack_feature: int
        self.attack_feature = attack_feature
        self._check_params()
        self.estimator = classifier

    def _check_params(self) -> None:
        if self.attack_feature < 0:
            raise ValueError("Attack feature must be positive.")
        
    def infer(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Infer the attacked feature.

        If the model's prediction coincides with the real prediction for the sample for a single value, choose it as the
        predicted value. If not, fall back to the Fredrikson method (without phi)

        Parameters
        ----------
        x : np.ndarray
            Input to attack. Includes all features except the attacked feature.
        y : np.ndarray
            Original model's predictions for x.
        values : list
            Possible values for attacked feature.
        priors : list
            Prior distributions of attacked feature values. Same size array as `values`.

        Returns
        -------
        np.ndarray
            The inferred feature values.
        """
        if "priors" not in kwargs:  # pragma: no cover
            raise ValueError("Missing parameter `priors`.")
        if "values" not in kwargs:  # pragma: no cover
            raise ValueError("Missing parameter `values`.")
        priors: Optional[list] = kwargs.get("priors")
        values: Optional[list] = kwargs.get("values")

        if priors is None or values is None:  # pragma: no cover
            raise ValueError("`priors` and `values` are required as inputs.")
        if len(priors) != len(values):  # pragma: no cover
            raise ValueError("Number of priors does not match number of values")

        n_values = len(values)
        n_samples = x.shape[0]

        # Will contain the model's predictions for each value
        pred_values = []
        # Will contain the probability of each value
        prob_values = []

        for i, value in enumerate(values):
            # prepare data with the given value in the attacked feature
            v_full = np.full((n_samples, 1), value).astype(x.dtype)
            x_value = np.concatenate((x[:, : self.attack_feature], v_full), axis=1)
            x_value = np.concatenate((x_value, x[:, self.attack_feature :]), axis=1)

            # Obtain the model's prediction for each possible value of the attacked feature
            pred_value = [np.argmax(arr) for arr in self.estimator.predict_proba(x_value)]
            pred_values.append(pred_value)

            # find the relative probability of this value for all samples being attacked
            prob_value = [
                (
                    (self.get_samples_at_node(self.get_decision_path([row])[-1]) / n_samples)
                    * priors[i]
                )
                for row in x_value
            ]
            prob_values.append(prob_value)

        # Find the single value that coincides with the real prediction for the sample (if it exists)
        pred_rows = zip(*pred_values)
        predicted_pred = []
        for row_index, row in enumerate(pred_rows):
            if y is not None:
                matches = [1 if row[value_index] == y[row_index] else 0 for value_index in range(n_values)]
                match_values = [
                    values[value_index] if row[value_index] == y[row_index] else 0 for value_index in range(n_values)
                ]
            else:
                matches = [0 for _ in range(n_values)]
                match_values = [0 for _ in range(n_values)]
            predicted_pred.append(sum(match_values) if sum(matches) == 1 else None)

        # Choose the value with highest probability for each sample
        predicted_prob = [np.argmax(list(prob)) for prob in zip(*prob_values)]

        return np.array(
            [
                value if value is not None else values[predicted_prob[index]]
                for index, value in enumerate(predicted_pred)
            ]
        )
    
    def get_decision_path(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the path through nodes in the tree when classifying x. Last one is leaf, first one root node.

        Parameters
        ----------
        x : np.ndarray
            Input sample.

        Returns
        -------
        np.ndarray
            The indices of the nodes in the array structure of the tree.
        """
        if len(np.shape(x)) == 1:
            return self.estimator.decision_path(x.reshape(1, -1)).indices

        return self.estimator.decision_path(x).indices
    
    def get_samples_at_node(self, node_id: int) -> int:
        """
        Returns the number of training samples mapped to a node.

        Parameters
        ----------
        node_id : int
            Node id.

        Returns
        -------
        int
            Number of samples mapped this node.
        """
        return self.estimator.tree_.n_node_samples[node_id]