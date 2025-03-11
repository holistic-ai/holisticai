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
This module implements the classifiers for scikit-learn models.
"""

from __future__ import annotations

import importlib
import logging
import os
import pickle
from copy import deepcopy
from typing import Callable, Optional, Union

import numpy as np
from holisticai.security.attackers.attribute_inference.mitigation import config
from holisticai.security.attackers.attribute_inference.mitigation.utils.formatting import to_categorical
from holisticai.security.attackers.attribute_inference.wrappers.classification.classifier import (
    ClassGradientsMixin,
    ClassifierMixin,
)
from holisticai.security.attackers.attribute_inference.wrappers.estimator import DecisionTreeMixin, LossGradientsMixin
from holisticai.security.attackers.attribute_inference.wrappers.regression.scikitlearn import (
    ScikitlearnDecisionTreeRegressor,
)
from holisticai.security.attackers.attribute_inference.wrappers.scikitlearn import ScikitlearnEstimator

logger = logging.getLogger(__name__)


def SklearnClassifier(  # noqa: N802
    model,
    clip_values=None,
    preprocessing_defences=None,
    postprocessing_defences=None,
    preprocessing=(0.0, 1.0),
    use_logits: bool = False,
):
    """
    Create a `Classifier` instance from a scikit-learn Classifier model. This is a convenience function that
    instantiates the correct class for the given scikit-learn model.

    Parameters
    ----------
    model : scikit-learn Classifier model
        The scikit-learn Classifier model.
    clip_values : tuple
        Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
    preprocessing_defences : `Preprocessor` or `list`
        Preprocessing defence(s) to be applied by the classifier.
    postprocessing_defences : `Postprocessor` or `list`
        Postprocessing defence(s) to be applied by the classifier.
    preprocessing : tuple
        Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be used for data
        preprocessing. The first value will be subtracted from the input. The input will then be divided by the
        second one.

    Returns
    -------
    Classifier
        The `Classifier` instance.
    """
    if model.__class__.__module__.split(".")[0] != "sklearn":  # pragma: no cover
        raise TypeError(f"Model is not an sklearn model. Received '{model.__class__}'")

    sklearn_name = model.__class__.__name__
    module = importlib.import_module(
        "holisticai.security.attackers.attribute_inference.wrappers.classification.scikitlearn"
    )
    if hasattr(module, f"Scikitlearn{sklearn_name}"):
        return getattr(module, f"Scikitlearn{sklearn_name}")(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

    # This basic class at least generically handles `fit`, `predict` and `save`
    return ScikitlearnClassifier(
        model,
        clip_values,
        preprocessing_defences,
        postprocessing_defences,
        preprocessing,
        use_logits,
    )


class ScikitlearnClassifier(ClassifierMixin, ScikitlearnEstimator):  # lgtm [py/missing-call-to-init]
    """
    Class for scikit-learn classifier models. Create a `Classifier` instance from a scikit-learn classifier model.

    Parameters
    ----------
    model : scikit-learn classifier model
        The scikit-learn classifier model.
    clip_values : tuple
        Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
    preprocessing_defences : `Preprocessor` or `list`
        Preprocessing defence(s) to be applied by the classifier.
    postprocessing_defences : `Postprocessor` or `list`
        Postprocessing defence(s) to be applied by the classifier.
    preprocessing : tuple
        Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be used for data
        preprocessing. The first value will be subtracted from the input. The input will then be divided by the
        second one.
    use_logits : bool
        Determines whether predict() returns logits instead of probabilities if available. Some adversarial attacks
        (DeepFool) may perform better if logits are used.
    """

    estimator_params = ClassifierMixin.estimator_params + ScikitlearnEstimator.estimator_params + ["use_logits"]

    def __init__(
        self,
        model,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0.0, 1.0),
        use_logits: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self._input_shape = self._get_input_shape(model)
        nb_classes = self._get_nb_classes()
        if nb_classes is not None:
            self.nb_classes = nb_classes
        self._use_logits = use_logits

    @property
    def input_shape(self) -> tuple[int, ...]:
        """
        Return the shape of one input sample.

        Returns
        -------
        tuple
            Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @property
    def use_logits(self) -> bool:
        """
        Return the Boolean for using logits.

        Returns
        -------
        bool
            Boolean for using logits.
        """
        return self._use_logits  # type: ignore

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        Parameters
        ----------
        x : `np.ndarray`
            Training data.
        y : `np.ndarray`
            Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        kwargs : `dict`
            Dictionary of framework-specific arguments. These should be parameters supported by the `fit` function in
            `sklearn` classifier and will be passed to this function as such.
        """
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        self.model.fit(x_preprocessed, y_preprocessed, **kwargs)
        self._input_shape = self._get_input_shape(self.model)
        self.nb_classes = self._get_nb_classes()

    def predict_proba(self, x: np.ndarray, **kwargs) -> np.ndarray:  # noqa: ARG002
        """
        Perform prediction for a batch of inputs.

        Parameters
        ----------
        x : `np.ndarray`
            Input samples.
        kwargs : `dict`
            Dictionary of framework-specific arguments. These should be parameters supported by the `predict_proba`
            function in `sklearn` classifier and will be passed to this function as such.

        Raises
        ------
        ValueError
            If the model does not have methods `predict_proba` or `predict`.

        Returns
        -------
        `np.ndarray`
            Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        if self._use_logits:
            if callable(getattr(self.model, "predict_log_proba", None)):
                y_pred = self.model.predict_log_proba(x_preprocessed)
            else:  # pragma: no cover
                logger.warning(
                    "use_logits was True but classifier did not have callable predict_log_proba member. Falling back to"
                    " probabilities"
                )
        elif callable(getattr(self.model, "predict_proba", None)):
            y_pred = self.model.predict_proba(x_preprocessed)
        elif callable(getattr(self.model, "predict", None)):
            y_pred = to_categorical(
                self.model.predict(x_preprocessed),
                nb_classes=self.model.classes_.shape[0],
            )
        else:  # pragma: no cover
            raise ValueError("The provided model does not have methods `predict_proba` or `predict`.")

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=y_pred, fit=False)

        return predictions

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        Parameters
        ----------
        x : `np.ndarray`
            Input samples.
        kwargs : `dict`
            Dictionary of framework-specific arguments. These should be parameters supported by the `predict` function in
            `sklearn` classifier and will be passed to this function as such.

        Returns
        -------
        `np.ndarray`
            Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        predictions = self.predict_proba(x=x, **kwargs)
        predictions = np.argmax(predictions, axis=1)
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
        if not os.path.exists(folder):  # pragma: no cover
            os.makedirs(folder)

        with open(full_path + ".pickle", "wb") as file_pickle:
            pickle.dump(self.model, file=file_pickle)

    def clone_for_refitting(self):
        """
        Create a copy of the classifier that can be refit from scratch.

        Returns
        -------
        `Classifier`
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

    def _get_nb_classes(self) -> int:
        if hasattr(self.model, "n_classes_"):
            _nb_classes = self.model.n_classes_
        elif hasattr(self.model, "classes_"):
            _nb_classes = self.model.classes_.shape[0]
        elif hasattr(self.model, "steps"):
            _nb_classes = self.model.steps[-1][1].obj.nb_classes
        else:
            logger.warning("Number of classes not recognised. The model might not have been fitted.")
            _nb_classes = None
        return _nb_classes


class ScikitlearnDecisionTreeClassifier(ScikitlearnClassifier):
    """
    Create a `Classifier` instance from a scikit-learn Decision Tree Classifier model.

    Parameters
    ----------
    model : scikit-learn Decision Tree Classifier model
        The scikit-learn Decision Tree Classifier model.
    clip_values : tuple
        Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
    preprocessing_defences : `Preprocessor` or `list`
        Preprocessing defence(s) to be applied by the classifier.
    postprocessing_defences : `Postprocessor` or `list`
        Postprocessing defence(s) to be applied by the classifier.
    preprocessing : tuple
        Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be used for data
        preprocessing. The first value will be subtracted from the input. The input will then be divided by the
        second one.
    """

    def __init__(
        self,
        model,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0.0, 1.0),
    ) -> None:
        import sklearn  # lgtm [py/repeated-import]
        from holisticai.pipeline import Pipeline

        if isinstance(model, Pipeline):
            if not isinstance(model.estimator_hdl.estimator, sklearn.tree.DecisionTreeClassifier) and model is not None:
                raise TypeError("Model must be of type sklearn.tree.DecisionTreeClassifier.")

        elif not isinstance(model, sklearn.tree.DecisionTreeClassifier) and model is not None:
            raise TypeError("Model must be of type sklearn.tree.DecisionTreeClassifier.")

        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

    def get_classes_at_node(self, node_id: int) -> np.ndarray:
        """
        Returns the classification for a given node.

        Parameters
        ----------
        node_id : int
            Node id.

        Returns
        -------
        np.ndarray
            Classification for the node.
        """
        return np.argmax(self.model.tree_.value[node_id])  # type: ignore

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
        x : np.ndarray
            Input sample.

        Returns
        -------
        np.ndarray
            The indices of the nodes in the array structure of the tree.
        """
        if len(np.shape(x)) == 1:
            return self.model.decision_path(x.reshape(1, -1)).indices

        return self.model.decision_path(x).indices

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
        return self.model.tree_.value[node_id] / np.linalg.norm(self.model.tree_.value[node_id])

    def _get_leaf_nodes(self, node_id, i_tree, class_label, box):
        from holisticai.security.attackers.attribute_inference.mitigation.verification_decisions_trees import (
            Box,
            Interval,
            LeafNode,
        )

        leaf_nodes = []

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
                    value=self.get_values_at_node(node_id)[0, class_label],
                )
            )

        return leaf_nodes


class ScikitlearnExtraTreeClassifier(ScikitlearnDecisionTreeClassifier):
    """
    Class for scikit-learn Extra TreeClassifier Classifier models.

    Parameters
    ----------
    model : scikit-learn Extra TreeClassifier Classifier model
        The scikit-learn Extra TreeClassifier Classifier model.
    clip_values : tuple
        Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
    preprocessing_defences : `Preprocessor` or `list`
        Preprocessing defence(s) to be applied by the classifier.
    postprocessing_defences : `Postprocessor` or `list`
        Postprocessing defence(s) to be applied by the classifier.
    preprocessing : tuple
        Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be used for data
        preprocessing. The first value will be subtracted from the input. The input will then be divided by the
        second one.
    """

    def __init__(
        self,
        model,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0.0, 1.0),
    ) -> None:
        import sklearn  # lgtm [py/repeated-import]

        if not isinstance(model, sklearn.tree.ExtraTreeClassifier):
            raise TypeError("Model must be of type sklearn.tree.ExtraTreeClassifier.")

        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )


class ScikitlearnAdaBoostClassifier(ScikitlearnClassifier):
    """
    Class for scikit-learn AdaBoost Classifier models.

    Parameters
    ----------
    model : scikit-learn AdaBoost Classifier model
        The scikit-learn AdaBoost Classifier model.
    clip_values : tuple
        Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
    preprocessing_defences : `Preprocessor` or `list`
        Preprocessing defence(s) to be applied by the classifier.
    postprocessing_defences : `Postprocessor` or `list`
        Postprocessing defence(s) to be applied by the classifier.
    preprocessing : tuple
        Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be used for data
        preprocessing. The first value will be subtracted from the input. The input will then be divided by the
        second one.
    """

    def __init__(
        self,
        model,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0.0, 1.0),
    ) -> None:
        import sklearn  # lgtm [py/repeated-import]

        if not isinstance(model, sklearn.ensemble.AdaBoostClassifier):
            raise TypeError("Model must be of type sklearn.ensemble.AdaBoostClassifier.")

        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )


class ScikitlearnBaggingClassifier(ScikitlearnClassifier):
    """
    Class for scikit-learn Bagging Classifier models.

    Parameters
    ----------
    model : scikit-learn Bagging Classifier model
        The scikit-learn Bagging Classifier model.
    clip_values : tuple
        Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
    preprocessing_defences : `Preprocessor` or `list`
        Preprocessing defence(s) to be applied by the classifier.
    postprocessing_defences : `Postprocessor` or `list`
        Postprocessing defence(s) to be applied by the classifier.
    preprocessing : tuple
        Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be used for data
        preprocessing. The first value will be subtracted from the input. The input will then be divided by the
        second one.
    """

    def __init__(
        self,
        model,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0.0, 1.0),
    ) -> None:
        import sklearn  # lgtm [py/repeated-import]

        if not isinstance(model, sklearn.ensemble.BaggingClassifier):
            raise TypeError("Model must be of type sklearn.ensemble.BaggingClassifier.")

        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )


class ScikitlearnExtraTreesClassifier(ScikitlearnClassifier, DecisionTreeMixin):
    """
    Class for scikit-learn Extra Trees Classifier models.

    Parameters
    ----------
    model : scikit-learn Extra Trees Classifier model
        The scikit-learn Extra Trees Classifier model.
    clip_values : tuple
        Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
    preprocessing_defences : `Preprocessor` or `list`
        Preprocessing defence(s) to be applied by the classifier.
    postprocessing_defences : `Postprocessor` or `list`
        Postprocessing defence(s) to be applied by the classifier.
    preprocessing : tuple
        Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be used for data
        preprocessing. The first value will be subtracted from the input. The input will then be divided by the
        second one.
    """

    def __init__(
        self,
        model,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0.0, 1.0),
    ):
        import sklearn  # lgtm [py/repeated-import]

        if not isinstance(model, sklearn.ensemble.ExtraTreesClassifier):
            raise TypeError("Model must be of type sklearn.ensemble.ExtraTreesClassifier.")

        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

    def get_trees(self):  # lgtm [py/similar-function]
        """
        Get the decision trees.

        Returns
        -------
        list
            A list of decision trees.
        """
        from holisticai.security.attackers.attribute_inference.mitigation.verification_decisions_trees import Box, Tree

        trees = []

        for i_tree, decision_tree_model in enumerate(self.model.estimators_):
            box = Box()

            extra_tree_classifier = ScikitlearnExtraTreeClassifier(model=decision_tree_model)

            for i_class in range(self.model.n_classes_):
                class_label = i_class

                trees.append(
                    Tree(
                        class_id=class_label,
                        leaf_nodes=extra_tree_classifier._get_leaf_nodes(0, i_tree, class_label, box),  # noqa: SLF001
                    )
                )

        return trees


class ScikitlearnGradientBoostingClassifier(ScikitlearnClassifier, DecisionTreeMixin):
    """
    Class for scikit-learn Gradient Boosting Classifier models.

    Parameters
    ----------
    model : scikit-learn Gradient Boosting Classifier model
        The scikit-learn Gradient Boosting Classifier model.
    clip_values : tuple
        Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
    preprocessing_defences : `Preprocessor` or `list`
        Preprocessing defence(s) to be applied by the classifier.
    postprocessing_defences : `Postprocessor` or `list`
        Postprocessing defence(s) to be applied by the classifier.
    preprocessing : tuple
        Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be used for data
        preprocessing. The first value will be subtracted from the input. The input will then be divided by the
        second one.
    """

    def __init__(
        self,
        model,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0.0, 1.0),
    ) -> None:
        import sklearn

        if not isinstance(model, sklearn.ensemble.GradientBoostingClassifier):
            raise TypeError("Model must be of type sklearn.ensemble.GradientBoostingClassifier.")

        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

    def get_trees(self):
        """
        Get the decision trees.

        Returns
        -------
        list
            A list of decision trees.
        """
        from holisticai.security.attackers.attribute_inference.mitigation.verification_decisions_trees import Box, Tree

        trees = []
        num_trees, num_classes = self.model.estimators_.shape

        for i_tree in range(num_trees):
            box = Box()

            for i_class in range(num_classes):
                decision_tree_classifier = ScikitlearnDecisionTreeRegressor(
                    model=self.model.estimators_[i_tree, i_class]
                )

                class_label = None if num_classes == 2 else i_class

                trees.append(
                    Tree(
                        class_id=class_label,
                        leaf_nodes=decision_tree_classifier._get_leaf_nodes(0, i_tree, class_label, box),  # noqa: SLF001
                    )
                )

        return trees


class ScikitlearnRandomForestClassifier(ScikitlearnClassifier):
    """
    Class for scikit-learn Random Forest Classifier models.

    Parameters
    ----------
    model : scikit-learn Random Forest Classifier model
        The scikit-learn Random Forest Classifier model.
    clip_values : tuple of float
        Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
    preprocessing_defences : `Preprocessor` or `list`
        Preprocessing defence(s) to be applied by the classifier.
    postprocessing_defences : `Postprocessor` or `list`
        Postprocessing defence(s) to be applied by the classifier.
    preprocessing : tuple of float
        Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be used for data
        preprocessing. The first value will be subtracted from the input. The input will then be divided by the
        second one.
    """

    def __init__(
        self,
        model,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0.0, 1.0),
    ) -> None:
        import sklearn

        if not isinstance(model, sklearn.ensemble.RandomForestClassifier):
            raise TypeError("Model must be of type sklearn.ensemble.RandomForestClassifier.")

        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

    def get_trees(self):  # lgtm [py/similar-function]
        """
        Get the decision trees.

        :return: A list of decision trees.
        """
        from holisticai.security.attackers.attribute_inference.mitigation.verification_decisions_trees import Box, Tree

        trees = []

        for i_tree, decision_tree_model in enumerate(self.model.estimators_):
            box = Box()

            decision_tree_classifier = ScikitlearnDecisionTreeClassifier(model=decision_tree_model)

            for i_class in range(self.model.n_classes_):
                class_label = i_class

                trees.append(
                    Tree(
                        class_id=class_label,
                        leaf_nodes=decision_tree_classifier._get_leaf_nodes(0, i_tree, class_label, box),  # noqa: SLF001
                    )
                )

        return trees


class ScikitlearnLogisticRegression(ClassGradientsMixin, LossGradientsMixin, ScikitlearnClassifier):
    """
    Class for scikit-learn Logistic Regression models.

    Parameters
    ----------
    model : scikit-learn Logistic Regression model
        The scikit-learn Logistic Regression model.
    clip_values : tuple
        Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
    preprocessing_defences : `Preprocessor` or `list`
        Preprocessing defence(s) to be applied by the classifier.
    postprocessing_defences : `Postprocessor` or `list`
        Postprocessing defence(s) to be applied by the classifier.
    preprocessing : tuple
        Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be used for data
        preprocessing. The first value will be subtracted from the input. The input will then be divided by the
        second one.
    """

    def __init__(
        self,
        model,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0.0, 1.0),
    ) -> None:
        import sklearn

        if not isinstance(model, sklearn.linear_model.LogisticRegression):
            raise TypeError("Model must be of type sklearn.linear_model.LogisticRegression).")

        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

    def class_gradient(self, x: np.ndarray, label: Union[int, list[int], None] = None, **kwargs) -> np.ndarray:  # noqa: ARG002
        """
        Compute per-class derivatives w.r.t. `x`.

        | Paper link: http://cs229.stanford.edu/proj2016/report/ItkinaWu-AdversarialAttacksonImageRecognition-report.pdf
        | Typo in https://arxiv.org/abs/1605.07277 (equation 6)

        Parameters
        ----------
        x : `np.ndarray`
            Sample input with shape as expected by the model.
        label : `int`, `list` of `int` or `None`
            Index of a specific per-class derivative. If an integer is provided, the gradient of that class output is
            computed for all samples. If multiple values as provided, the first dimension should match the batch size
            of `x`, and each value will be used as target for its corresponding sample in `x`. If `None`, then gradients
            for all classes will be computed for each sample.

        Returns
        -------
        `np.ndarray`
            Array of gradients of input features w.r.t. each class in the form `(batch_size, nb_classes, input_shape)`

        Raises
        ------
        ValueError
            If the model has not been fitted prior to calling this method or if the number of classes in the classifier
            is not known.
        TypeError
            If the requested label cannot be processed.
        """
        if not hasattr(self.model, "coef_"):  # pragma: no cover
            raise ValueError(
                """Model has not been fitted. Run function `fit(x, y)` of classifier first or provide a
            fitted model."""
            )
        if self.nb_classes is None:  # pragma: no cover
            raise ValueError("Unknown number of classes in classifier.")
        nb_samples = x.shape[0]

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        y_pred = self.model.predict_proba(X=x_preprocessed)
        weights = self.model.coef_

        if self.nb_classes > 2:  # type: ignore
            w_weighted = np.matmul(y_pred, weights)

        def _f_class_gradient(i_class, i_sample):
            if self.nb_classes == 2:
                return (-1.0) ** (i_class + 1.0) * y_pred[i_sample, 0] * y_pred[i_sample, 1] * weights[0, :]

            return weights[i_class, :] - w_weighted[i_sample, :]

        if label is None:
            # Compute the gradients w.r.t. all classes
            class_gradients = []

            for i_class in range(self.nb_classes):  # type: ignore
                class_gradient = np.zeros(x.shape)
                for i_sample in range(nb_samples):
                    class_gradient[i_sample, :] += _f_class_gradient(i_class, i_sample)
                class_gradients.append(class_gradient)

            gradients = np.swapaxes(np.array(class_gradients), 0, 1)

        elif isinstance(label, int):
            # Compute the gradients only w.r.t. the provided label
            class_gradient = np.zeros(x.shape)
            for i_sample in range(nb_samples):
                class_gradient[i_sample, :] += _f_class_gradient(label, i_sample)

            gradients = np.swapaxes(np.array([class_gradient]), 0, 1)

        elif (
            (isinstance(label, list) and len(label) == nb_samples)
            or isinstance(label, np.ndarray)
            and label.shape == (nb_samples,)
        ):
            # For each sample, compute the gradients w.r.t. the indicated target class (possibly distinct)
            class_gradients = []
            unique_labels = list(np.unique(label))

            for unique_label in unique_labels:
                class_gradient = np.zeros(x.shape)
                for i_sample in range(nb_samples):
                    # class_gradient[i_sample, :] += label[i_sample, unique_label] * (weights[unique_label, :]
                    # - w_weighted[i_sample, :])
                    class_gradient[i_sample, :] += _f_class_gradient(unique_label, i_sample)

                class_gradients.append(class_gradient)

            gradients = np.swapaxes(np.array(class_gradients), 0, 1)
            lst = [unique_labels.index(i) for i in label]
            gradients = np.expand_dims(gradients[np.arange(len(gradients)), lst], axis=1)

        else:
            raise TypeError("Unrecognized type for argument `label` with type " + str(type(label)))

        gradients = self._apply_preprocessing_gradient(x, gradients)

        return gradients

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:  # noqa: ARG002
        """
        Compute the gradient of the loss function w.r.t. `x`.

        Parameters
        ----------
        x : `np.ndarray`
            Sample input with shape as expected by the model.
        y : `np.ndarray`
            Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        kwargs : `dict`
            Dictionary of framework-specific arguments. These should be parameters supported by the `loss_gradient`
            function in `sklearn` classifier and will be passed to this function as such.

        Returns
        -------
        `np.ndarray`
            Array of gradients of the same shape as `x`.

        Raises
        ------
        ValueError
            If the model has not been fitted prior to calling this method.
        """
        from sklearn.utils.class_weight import compute_class_weight

        if not hasattr(self.model, "coef_"):  # pragma: no cover
            raise ValueError(
                """Model has not been fitted. Run function `fit(x, y)` of classifier first or provide a
            fitted model."""
            )

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        y_index = np.argmax(y_preprocessed, axis=1)
        if self.model.class_weight is None or self.model.class_weight == "balanced":
            class_weight = np.ones(self.nb_classes)
        else:
            class_weight = compute_class_weight(
                class_weight=self.model.class_weight,
                classes=self.model.classes_,
                y=y_index,
            )

        y_pred = self.predict(x=x_preprocessed)
        weights = self.model.coef_

        errors = class_weight * (y_pred - y)

        if weights.shape[0] == 1:
            weights = np.append(-weights, weights, axis=0)

        gradients = (errors @ weights) / self.model.classes_.size

        gradients = self._apply_preprocessing_gradient(x, gradients)

        return gradients

    @staticmethod
    def get_trainable_attribute_names() -> tuple[str, str]:
        """
        Get the names of trainable attributes.

        Returns
        -------
        `tuple`
            A tuple of trainable attributes.
        """
        return "intercept_", "coef_"


class ScikitlearnGaussianNB(ScikitlearnClassifier):
    """
    Class for scikit-learn Gaussian Naive Bayes models.

    Parameters
    ----------
    model : scikit-learn Gaussian Naive Bayes model
        The scikit-learn Gaussian Naive Bayes model.
    clip_values : tuple
        Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
    preprocessing_defences : `Preprocessor` or `list`
        Preprocessing defence(s) to be applied by the classifier.
    postprocessing_defences : `Postprocessor` or `list`
        Postprocessing defence(s) to be applied by the classifier.
    preprocessing : tuple
        Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be used for data
        preprocessing. The first value will be subtracted from the input. The input will then be divided by the
        second one.
    """

    def __init__(
        self,
        model,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0.0, 1.0),
    ) -> None:
        import sklearn  # lgtm [py/repeated-import]

        if not isinstance(model, sklearn.naive_bayes.GaussianNB):  # pragma: no cover
            raise TypeError(f"Model must be of type sklearn.naive_bayes.GaussianNB. Found type {type(model)}")

        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

    @staticmethod
    def get_trainable_attribute_names() -> tuple[str, str]:
        """
        Get the names of trainable attributes.

        Returns
        -------
        `tuple`
            A tuple of trainable attributes.
        """
        return "sigma_", "theta_"


class ScikitlearnSVC(ClassGradientsMixin, LossGradientsMixin, ScikitlearnClassifier):
    """
    Class for scikit-learn C-Support Vector Classification models.

    Parameters
    ----------
    model : scikit-learn C-Support Vector Classification model
        The scikit-learn C-Support Vector Classification model.
    clip_values : tuple
        Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
    preprocessing_defences : `Preprocessor` or `list`
        Preprocessing defence(s) to be applied by the classifier.
    postprocessing_defences : `Postprocessor` or `list`
        Postprocessing defence(s) to be applied by the classifier.
    preprocessing : tuple
        Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be used for data
        preprocessing. The first value will be subtracted from the input. The input will then be divided by the
        second one.
    """

    def __init__(
        self,
        model,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0.0, 1.0),
    ) -> None:
        import sklearn  # lgtm [py/repeated-import]

        if not isinstance(model, sklearn.svm.SVC) and not isinstance(model, sklearn.svm.LinearSVC):
            raise TypeError(f"Model must be of type sklearn.svm.SVC or sklearn.svm.LinearSVC. Found type {type(model)}")

        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self._kernel = self._kernel_func()

    def class_gradient(self, x: np.ndarray, label: Union[int, list[int], None] = None, **kwargs) -> np.ndarray:  # noqa: ARG002
        """
        Compute per-class derivatives w.r.t. `x`.

        Parameters
        ----------
        x : `np.ndarray`
            Sample input with shape as expected by the model.
        label : `int`, `list` of `int` or `None`
            Index of a specific per-class derivative. If an integer is provided, the gradient of that class output is
            computed for all samples. If multiple values as provided, the first dimension should match the batch size
            of `x`, and each value will be used as target for its corresponding sample in `x`. If `None`, then gradients
            for all classes will be computed for each sample.

        Returns
        -------
        `np.ndarray`
            Array of gradients of input features w.r.t. each class in the form `(batch_size, nb_classes, input_shape)`
        """
        import sklearn  # lgtm [py/repeated-import]

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        num_samples, _ = x_preprocessed.shape

        if isinstance(self.model, sklearn.svm.SVC):
            if self.model.fit_status_:  # pragma: no cover
                raise AssertionError("Model has not been fitted correctly.")

            support_indices = [0, *list(np.cumsum(self.model.n_support_))]

            sign_multiplier = -1 if self.nb_classes == 2 else 1

            if label is None:
                gradients = np.zeros(
                    (
                        x_preprocessed.shape[0],
                        self.nb_classes,
                        x_preprocessed.shape[1],
                    )
                )

                for i_label in range(self.nb_classes):  # type: ignore
                    for i_sample in range(num_samples):
                        for not_label in range(self.nb_classes):  # type: ignore
                            if i_label != not_label:
                                label_multiplier = -1 if not_label < i_label else 1

                                for label_sv in range(
                                    support_indices[i_label],
                                    support_indices[i_label + 1],
                                ):
                                    alpha_i_k_y_i = self.model.dual_coef_[
                                        not_label if not_label < i_label else not_label - 1,
                                        label_sv,
                                    ]
                                    grad_kernel = self._get_kernel_gradient_sv(label_sv, x_preprocessed[i_sample])
                                    gradients[i_sample, i_label] += label_multiplier * alpha_i_k_y_i * grad_kernel

                                for not_label_sv in range(
                                    support_indices[not_label],
                                    support_indices[not_label + 1],
                                ):
                                    alpha_i_k_y_i = self.model.dual_coef_[
                                        i_label if i_label < not_label else i_label - 1,
                                        not_label_sv,
                                    ]
                                    grad_kernel = self._get_kernel_gradient_sv(not_label_sv, x_preprocessed[i_sample])
                                    gradients[i_sample, i_label] += label_multiplier * alpha_i_k_y_i * grad_kernel

            elif isinstance(label, int):
                gradients = np.zeros((x_preprocessed.shape[0], 1, x_preprocessed.shape[1]))

                for i_sample in range(num_samples):
                    for not_label in range(self.nb_classes):  # type: ignore
                        if label != not_label:
                            label_multiplier = -1 if not_label < label else 1

                            for label_sv in range(support_indices[label], support_indices[label + 1]):
                                alpha_i_k_y_i = self.model.dual_coef_[
                                    not_label if not_label < label else not_label - 1,
                                    label_sv,
                                ]
                                grad_kernel = self._get_kernel_gradient_sv(label_sv, x_preprocessed[i_sample])
                                gradients[i_sample, 0] += label_multiplier * alpha_i_k_y_i * grad_kernel

                            for not_label_sv in range(
                                support_indices[not_label],
                                support_indices[not_label + 1],
                            ):
                                alpha_i_k_y_i = self.model.dual_coef_[
                                    label if label < not_label else label - 1,
                                    not_label_sv,
                                ]
                                grad_kernel = self._get_kernel_gradient_sv(not_label_sv, x_preprocessed[i_sample])
                                gradients[i_sample, 0] += label_multiplier * alpha_i_k_y_i * grad_kernel

            elif (
                (isinstance(label, list) and len(label) == num_samples)
                or isinstance(label, np.ndarray)
                and label.shape == (num_samples,)
            ):
                gradients = np.zeros((x_preprocessed.shape[0], 1, x_preprocessed.shape[1]))

                for i_sample in range(num_samples):
                    for not_label in range(self.nb_classes):  # type: ignore
                        if label[i_sample] != not_label:
                            label_multiplier = -1 if not_label < label[i_sample] else 1

                            for label_sv in range(
                                support_indices[label[i_sample]],
                                support_indices[label[i_sample] + 1],
                            ):
                                alpha_i_k_y_i = self.model.dual_coef_[
                                    not_label if not_label < label[i_sample] else not_label - 1,
                                    label_sv,
                                ]
                                grad_kernel = self._get_kernel_gradient_sv(label_sv, x_preprocessed[i_sample])
                                gradients[i_sample, 0] += label_multiplier * alpha_i_k_y_i * grad_kernel

                            for not_label_sv in range(
                                support_indices[not_label],
                                support_indices[not_label + 1],
                            ):
                                alpha_i_k_y_i = self.model.dual_coef_[
                                    label[i_sample] if label[i_sample] < not_label else label[i_sample] - 1,
                                    not_label_sv,
                                ]
                                grad_kernel = self._get_kernel_gradient_sv(not_label_sv, x_preprocessed[i_sample])
                                gradients[i_sample, 0] += label_multiplier * alpha_i_k_y_i * grad_kernel

            else:
                raise TypeError("Unrecognized type for argument `label` with type " + str(type(label)))

            gradients = self._apply_preprocessing_gradient(x, gradients * sign_multiplier)

        elif isinstance(self.model, sklearn.svm.LinearSVC):
            if label is None:
                gradients = np.zeros(
                    (
                        x_preprocessed.shape[0],
                        self.nb_classes,
                        x_preprocessed.shape[1],
                    )
                )

                for i in range(self.nb_classes):  # type: ignore
                    for i_sample in range(num_samples):
                        if self.nb_classes == 2:
                            gradients[i_sample, i] = self.model.coef_[0] * (2 * i - 1)
                        else:
                            gradients[i_sample, i] = self.model.coef_[i]

            elif isinstance(label, int):
                gradients = np.zeros((x_preprocessed.shape[0], 1, x_preprocessed.shape[1]))

                for i_sample in range(num_samples):
                    if self.nb_classes == 2:
                        gradients[i_sample, 0] = self.model.coef_[0] * (2 * label - 1)
                    else:
                        gradients[i_sample, 0] = self.model.coef_[label]

            elif (
                (isinstance(label, list) and len(label) == num_samples)
                or isinstance(label, np.ndarray)
                and label.shape == (num_samples,)
            ):
                gradients = np.zeros((x_preprocessed.shape[0], 1, x_preprocessed.shape[1]))

                for i_sample in range(num_samples):
                    if self.nb_classes == 2:
                        gradients[i_sample, 0] = self.model.coef_[0] * (2 * label[i_sample] - 1)
                    else:
                        gradients[i_sample, 0] = self.model.coef_[label[i_sample]]

            else:
                raise TypeError("Unrecognized type for argument `label` with type " + str(type(label)))

            gradients = self._apply_preprocessing_gradient(x, gradients)

        return gradients

    def _kernel_grad(self, sv: np.ndarray, x_sample: np.ndarray) -> np.ndarray:
        """
        Applies the kernel gradient to a support vector.

        Parameters
        ----------
        sv : `np.ndarray`
            A support vector.
        x_sample : `np.ndarray`
            The sample the gradient is taken with respect to.

        Returns
        -------
        `np.ndarray`
            The kernel gradient.
        """
        if self.model.kernel == "linear":
            grad = sv
        elif self.model.kernel == "poly":
            grad = (
                self.model.degree
                * (self.model._gamma * np.sum(x_sample * sv) + self.model.coef0) ** (self.model.degree - 1)  # noqa: SLF001
                * sv
            )
        elif self.model.kernel == "rbf":
            grad = (
                2
                * self.model._gamma  # noqa: SLF001
                * (-1)
                * np.exp(-self.model._gamma * np.linalg.norm(x_sample - sv, ord=2) ** 2)  # noqa: SLF001
                * (x_sample - sv)
            )
        elif self.model.kernel == "sigmoid":
            raise NotImplementedError
        else:
            raise NotImplementedError(f"Loss gradients for kernel '{self.model.kernel}' are not implemented.")
        return grad

    def _get_kernel_gradient_sv(self, i_sv: int, x_sample: np.ndarray) -> np.ndarray:
        """
        Applies the kernel gradient to all of a model's support vectors.

        Parameters
        ----------
        i_sv : `int`
            A support vector index.
        x_sample : `np.ndarray`
            The sample the gradient is taken with respect to.

        Returns
        -------
        `np.ndarray`
            The kernel gradient.
        """
        x_i = self.model.support_vectors_[i_sv, :]
        return self._kernel_grad(x_i, x_sample)

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:  # noqa: ARG002
        """
        Compute the gradient of the loss function w.r.t. `x`.
        Following equation (1) with lambda=0.

        | Paper link: https://pralab.diee.unica.it/sites/default/files/biggio14-svm-chapter.pdf

        Parameters
        ----------
        x : `np.ndarray`
            Sample input with shape as expected by the model.
        y : `np.ndarray`
            Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        kwargs : `dict`
            Dictionary of framework-specific arguments. These should be parameters supported by the `loss_gradient`
            function in `sklearn` classifier and will be passed to this function as such.

        Returns
        -------
        `np.ndarray`
            Array of gradients of the same shape as `x`.

        Raises
        ------
        ValueError
            If the model has not been fitted prior to calling this method.

        """
        import sklearn  # lgtm [py/repeated-import]

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        num_samples, _ = x_preprocessed.shape
        gradients = np.zeros_like(x_preprocessed)
        y_index = np.argmax(y_preprocessed, axis=1)

        if isinstance(self.model, sklearn.svm.SVC):
            if self.model.fit_status_:
                raise AssertionError("Model has not been fitted correctly.")

            sign_multiplier = 1 if y_preprocessed.shape[1] == 2 else -1

            i_not_label_i = None
            label_multiplier = None
            support_indices = [0, *list(np.cumsum(self.model.n_support_))]

            for i_sample in range(num_samples):
                i_label = y_index[i_sample]

                for i_not_label in range(self.nb_classes):  # type: ignore
                    if i_label != i_not_label:
                        if i_not_label < i_label:
                            i_not_label_i = i_not_label
                            label_multiplier = -1
                        elif i_not_label > i_label:
                            i_not_label_i = i_not_label - 1
                            label_multiplier = 1

                        for i_label_sv in range(support_indices[i_label], support_indices[i_label + 1]):
                            alpha_i_k_y_i = self.model.dual_coef_[i_not_label_i, i_label_sv] * label_multiplier
                            grad_kernel = self._get_kernel_gradient_sv(i_label_sv, x_preprocessed[i_sample])
                            gradients[i_sample, :] += sign_multiplier * alpha_i_k_y_i * grad_kernel

                        for i_not_label_sv in range(
                            support_indices[i_not_label],
                            support_indices[i_not_label + 1],
                        ):
                            alpha_i_k_y_i = self.model.dual_coef_[i_not_label_i, i_not_label_sv] * label_multiplier
                            grad_kernel = self._get_kernel_gradient_sv(i_not_label_sv, x_preprocessed[i_sample])
                            gradients[i_sample, :] += sign_multiplier * alpha_i_k_y_i * grad_kernel

        elif isinstance(self.model, sklearn.svm.LinearSVC):
            for i_sample in range(num_samples):
                i_label = y_index[i_sample]
                if self.nb_classes == 2:
                    i_label_i = 0
                    if i_label == 0:
                        label_multiplier = 1
                    elif i_label == 1:
                        label_multiplier = -1
                    else:
                        raise ValueError("Label index not recognized because it is not 0 or 1.")
                else:
                    i_label_i = i_label
                    label_multiplier = -1

                gradients[i_sample] = label_multiplier * self.model.coef_[i_label_i]
        else:
            raise TypeError("Model not recognized.")

        gradients = self._apply_preprocessing_gradient(x, gradients)
        return gradients

    def _kernel_func(self) -> Callable:
        """
        Return the function for the kernel of this SVM.

        Returns
        -------
        `Callable`
            A callable kernel function.
        """
        import sklearn  # lgtm [py/repeated-import]
        from sklearn.metrics.pairwise import (
            linear_kernel,
            polynomial_kernel,
            rbf_kernel,
        )

        if isinstance(self.model, sklearn.svm.LinearSVC):
            kernel = "linear"
        elif isinstance(self.model, sklearn.svm.SVC):
            kernel = self.model.kernel
        else:
            raise NotImplementedError("SVM model not yet supported.")

        if kernel == "linear":
            kernel_func = linear_kernel
        elif kernel == "poly":
            kernel_func = polynomial_kernel
        elif kernel == "rbf":
            kernel_func = rbf_kernel
        elif callable(kernel):
            kernel_func = kernel
        else:
            raise NotImplementedError(f"Kernel '{kernel}' not yet supported.")

        return kernel_func

    def q_submatrix(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """
        Returns the q submatrix of this SVM indexed by the arrays at rows and columns.

        Parameters
        ----------
        rows : `np.ndarray`
            The row vectors.
        cols : `np.ndarray`
            The column vectors.

        Returns
        -------
        `np.ndarray`
            A submatrix of Q.
        """
        submatrix_shape = (rows.shape[0], cols.shape[0])
        y_row = self.model.predict(rows)
        y_col = self.model.predict(cols)
        y_row[y_row == 0] = -1
        y_col[y_col == 0] = -1
        q_rc = np.zeros(submatrix_shape)
        for row in range(q_rc.shape[0]):
            for col in range(q_rc.shape[1]):
                q_rc[row][col] = self._kernel([rows[row]], [cols[col]])[0][0] * y_row[row] * y_col[col]

        return q_rc

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:  # noqa: ARG002
        """
        Perform prediction for a batch of inputs.

        Parameters
        ----------
        x : `np.ndarray`
            Input samples.

        Returns
        -------
        `np.ndarray`
            Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        import sklearn  # lgtm [py/repeated-import]

        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        if isinstance(self.model, sklearn.svm.SVC) and self.model.probability:
            y_pred = self.model.predict_proba(X=x_preprocessed)
        else:
            y_pred_label = self.model.predict(X=x_preprocessed)
            targets = np.array(y_pred_label).reshape(-1)
            one_hot_targets = np.eye(self.nb_classes)[targets]
            y_pred = one_hot_targets

        return y_pred


ScikitlearnLinearSVC = ScikitlearnSVC
