# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
This module implements the regressors for scikit-learn models.
"""

from __future__ import annotations

import logging
import os
import pickle
from copy import deepcopy
from typing import Optional

import numpy as np
from holisticai.security.attackers.attribute_inference.mitigation import config
from holisticai.security.attackers.attribute_inference.wrappers.regression.regressor import RegressorMixin
from holisticai.security.attackers.attribute_inference.wrappers.scikitlearn import ScikitlearnEstimator

logger = logging.getLogger(__name__)


class ScikitlearnRegressor(RegressorMixin, ScikitlearnEstimator):  # lgtm [py/missing-call-to-init]
    """
    Wrapper class for scikit-learn regression models.

    This class supports all regression models from scikit-learn.

    Parameters
    ----------
    model : object
        scikit-learn regression model.
    clip_values : tuple(float, float)
        Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
    preprocessing_defences : `Preprocessor` or `List[Preprocessor]`
        Preprocessing defence(s) to be applied by the classifier.
    postprocessing_defences : `Postprocessor` or `List[Postprocessor]`
        Postprocessing defence(s) to be applied by the classifier.
    preprocessing : `tuple(float, float)` or `np.ndarray`
        Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be used for data preprocessing.
        The first value will be subtracted from the input. The input will then be divided by the second one.
    """

    estimator_params = ScikitlearnEstimator.estimator_params

    def __init__(
        self,
        model,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0.0, 1.0),
    ) -> None:
        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self._input_shape = self._get_input_shape(model)

    @property
    def input_shape(self) -> tuple[int, ...]:
        """
        Return the shape of one input sample.

        Returns
        -------
        np.ndarray
            Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the regressor on the training set `(x, y)`.

        Parameters
        ----------
        x : `np.ndarray`
            Training data.
        y : `np.ndarray`
            Target values.
        kwargs : `dict`
            Dictionary of framework-specific arguments. These should be parameters supported by the `fit` function in
            `sklearn` regressor and will be passed to this function as such.
        """
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        self.model.fit(x_preprocessed, y_preprocessed, **kwargs)
        self._input_shape = self._get_input_shape(self.model)

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:  # noqa: ARG002
        """
        Perform prediction for a batch of inputs.

        Parameters
        ----------
        x : `np.ndarray`
            Input samples.
        kwargs : `dict`
            Dictionary of framework-specific arguments. These should be parameters supported by the `predict` function in
            `sklearn` regressor and will be passed to this function as such.

        Returns
        -------
        `np.ndarray`
            Array of predictions.
        """
        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        if callable(getattr(self.model, "predict", None)):
            y_pred = self.model.predict(x_preprocessed)
        else:
            raise TypeError("The provided model does not have the method `predict`.")

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=y_pred, fit=False)

        return predictions

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """
        Save a model to file in the format specific to the backend framework.

        Parameters
        ----------
        filename : str
            Name of the file where to store the model.
        path : str, optional
            Path of the folder where to store the model. If no path is specified, the model will be stored in the default
            data location of the library `ART_DATA_PATH`.
        """
        full_path = os.path.join(config.ART_DATA_PATH, filename) if path is None else os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(full_path + ".pickle", "wb") as file_pickle:
            pickle.dump(self.model, file=file_pickle)

    def clone_for_refitting(self):  # lgtm [py/inheritance/incorrect-overridden-signature]
        """
        Create a copy of the classifier that can be refit from scratch.

        Returns
        -------
        `Regressor`
            New estimator.
        """
        import sklearn  # lgtm [py/repeated-import]

        clone = type(self)(sklearn.base.clone(self.model))
        params = self.get_params()
        del params["model"]
        clone.set_params(**params)
        return clone

    def reset(self) -> None:
        """
        Resets the weights of the classifier so that it can be refit from scratch.

        """
        # No need to do anything since scikitlearn models start from scratch each time fit() is called

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:  # noqa: ARG002
        """
        Compute the MSE loss of the regressor for samples `x`.

        Parameters
        ----------
        x : `np.ndarray`
            Input samples.
        y : `np.ndarray`
            Target values.

        Returns
        -------
        `np.ndarray`
            Loss values.
        """

        return (y - self.predict(x)) ** 2

    def compute_loss_from_predictions(self, pred: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:  # noqa: ARG002
        """
        Compute the MSE loss of the regressor for predictions `pred`.

        Parameters
        ----------
        pred : `np.ndarray`
            Model predictions.
        y : `np.ndarray`
            Target values.

        Returns
        -------
        `np.ndarray`
            Loss values.
        """

        return (y - pred) ** 2


class ScikitlearnDecisionTreeRegressor(ScikitlearnRegressor):
    """
    Wrapper class for scikit-learn Decision Tree Regressor models.

    This class supports all Decision Tree Regressor models from scikit-learn.

    Parameters
    ----------
    model : object
        scikit-learn Decision Tree Regressor model.
    clip_values : tuple(float, float)
        Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
    preprocessing_defences : `Preprocessor` or `List[Preprocessor]`
        Preprocessing defence(s) to be applied by the classifier.
    postprocessing_defences : `Postprocessor` or `List[Postprocessor]`
        Postprocessing defence(s) to be applied by the classifier.
    preprocessing : `tuple(float, float)` or `np.ndarray`
        Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be used for data preprocessing.
        The first value will be subtracted from the input. The input will then be divided by the second one.
    """

    def __init__(
        self,
        model,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0.0, 1.0),
    ) -> None:
        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self._model = model

    def get_values_at_node(self, node_id: int) -> np.ndarray:
        """
        Returns the feature of given id for a node.

        Parameters
        ----------
        node_id : int
            Node id.

        Returns
        -------
        np.ndarray
            Normalized values at node node_id.
        """
        return self.model.tree_.value[node_id]

    def get_left_child(self, node_id: int) -> int:
        """
        Returns the id of the left child node of node_id.

        Parameters
        ----------
        node_id : int
            Node id.

        Returns
        -------
        int
            The indices of the left child in the tree.
        """
        return self.model.tree_.children_left[node_id]

    def get_right_child(self, node_id: int) -> int:
        """
        Returns the id of the right child node of node_id.

        Parameters
        ----------
        node_id : int
            Node id.

        Returns
        -------
        int
            The indices of the right child in the tree.
        """
        return self.model.tree_.children_right[node_id]

    def get_decision_path(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the path through nodes in the tree when classifying x. Last one is leaf, first one root node.

        Parameters
        ----------
        x : `np.ndarray`
            Input sample.

        Returns
        -------
        np.ndarray
            The indices of the nodes in the array structure of the tree.
        """
        if len(np.shape(x)) == 1:
            return self.model.decision_path(x.reshape(1, -1)).indices

        return self.model.decision_path(x).indices

    def get_threshold_at_node(self, node_id: int) -> float:
        """
        Returns the threshold of given id for a node.

        Parameters
        ----------
        node_id : int
            Node id.

        Returns
        -------
        float
            Threshold value of feature split in this node.
        """
        return self.model.tree_.threshold[node_id]

    def get_feature_at_node(self, node_id: int) -> int:
        """
        Returns the feature of given id for a node.

        Parameters
        ----------
        node_id : int
            Node id.

        Returns
        -------
        int
            Feature index of feature split in this node.
        """
        return self.model.tree_.feature[node_id]

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
        return self.model.tree_.n_node_samples[node_id]

    def _get_leaf_nodes(self, node_id, i_tree, class_label, box):
        from holisticai.security.attackers.attribute_inference.mitigation.verification_decisions_trees import (
            Box,
            Interval,
            LeafNode,
        )

        leaf_nodes: list[LeafNode] = []

        if self.get_left_child(node_id) != self.get_right_child(node_id):
            node_left = self.get_left_child(node_id)
            node_right = self.get_right_child(node_id)

            box_left = deepcopy(box)
            box_right = deepcopy(box)

            feature = self.get_feature_at_node(node_id)
            box_split_left = Box(intervals={feature: Interval(-np.inf, self.get_threshold_at_node(node_id))})
            box_split_right = Box(intervals={feature: Interval(self.get_threshold_at_node(node_id), np.inf)})

            if box.intervals:
                box_left.intersect_with_box(box_split_left)
                box_right.intersect_with_box(box_split_right)
            else:
                box_left = box_split_left
                box_right = box_split_right

            leaf_nodes += self._get_leaf_nodes(node_left, i_tree, class_label, box_left)
            leaf_nodes += self._get_leaf_nodes(node_right, i_tree, class_label, box_right)

        else:
            leaf_nodes.append(
                LeafNode(
                    tree_id=i_tree,
                    class_label=class_label,
                    node_id=node_id,
                    box=box,
                    value=self.get_values_at_node(node_id)[0, 0],
                )
            )

        return leaf_nodes
