from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.preprocessing import minmax_scale


def to_categorical(labels, nb_classes) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.

    Parameters
    ----------
    labels : np.ndarray
        An array of integer labels of shape `(nb_samples,)`.
    nb_classes : int
        The number of classes (possible labels).

    Returns
    -------
    np.ndarray
        A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=int)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical


def check_and_transform_label_format(
    labels: np.ndarray, nb_classes: Optional[int], return_one_hot: bool = True
) -> np.ndarray:
    """
    Check label format and transform to one-hot-encoded labels if necessary

    Parameters
    ----------
    labels : np.ndarray
        An array of integer labels of shape `(nb_samples,)`, `(nb_samples, 1)` or `(nb_samples, nb_classes)`.
    nb_classes : int
        The number of classes. If None the number of classes is determined automatically.
    return_one_hot : bool
        True if returning one-hot encoded labels, False if returning index labels.

    Returns
    -------
    np.ndarray
        Labels with shape `(nb_samples, nb_classes)` (one-hot) or `(nb_samples,)` (index).
    """
    labels_return = labels

    if len(labels.shape) == 2 and labels.shape[1] > 1:  # multi-class, one-hot encoded
        if not return_one_hot:
            labels_return = np.argmax(labels, axis=1)
            labels_return = np.expand_dims(labels_return, axis=1)
    elif len(labels.shape) == 2 and labels.shape[1] == 1:
        if nb_classes is None:
            nb_classes = np.max(labels) + 1
        if nb_classes > 2:  # multi-class, index labels
            labels_return = to_categorical(labels, nb_classes) if return_one_hot else np.expand_dims(labels, axis=1)
        elif nb_classes == 2:  # binary, index labels
            if return_one_hot:
                labels_return = to_categorical(labels, nb_classes)
        else:
            raise ValueError(
                "Shape of labels not recognised."
                "Please provide labels in shape (nb_samples,) or (nb_samples, "
                "nb_classes)"
            )
    elif len(labels.shape) == 1:  # index labels
        labels_return = to_categorical(labels, nb_classes) if return_one_hot else np.expand_dims(labels, axis=1)
    else:
        raise ValueError(
            "Shape of labels not recognised."
            "Please provide labels in shape (nb_samples,) or (nb_samples, "
            "nb_classes)"
        )

    return labels_return


def get_feature_values(x: np.ndarray, single_index_feature: bool) -> list:
    """
    Returns a list of unique values of a given feature.

    Parameters
    ----------
    x : np.ndarray
        The feature column(s).
    single_index_feature : bool
        Bool representing whether this is a single-column or multiple-column feature (for
        example 1-hot encoded and then scaled).

    Returns
    -------
    list
        For a single-column feature, a simple list containing all possible values, in increasing order.
        For a multi-column feature, a list of lists, where each internal list represents a column and the values
        represent the possible values for that column (in increasing order).
    """
    values = None
    if single_index_feature:
        values = np.unique(x).tolist()
    else:
        for column in x.T:
            column_values = np.unique(column)
            values = column_values if values is None else np.vstack((values, column_values))
        if values is not None:
            values = values.tolist()
    return values


def floats_to_one_hot(labels: np.ndarray):
    """
    Convert a 2D-array of floating point labels to binary class matrix.

    Parameters
    ----------
    labels : np.ndarray
        A 2D-array of floating point labels of shape `(nb_samples, nb_classes)`.

    Returns
    -------
    np.ndarray
        A binary matrix representation of `labels` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels)
    for feature in labels.T:  # pylint: disable=E1133
        unique = np.unique(feature)
        unique.sort()
        for index, value in enumerate(unique):
            feature[feature == value] = index
    return labels.astype(np.float32)


def float_to_categorical(labels: np.ndarray, nb_classes: Optional[int] = None):
    """
    Convert an array of floating point labels to binary class matrix.

    Parameters
    ----------
    labels : np.ndarray
        An array of floating point labels of shape `(nb_samples,)`.
    nb_classes : int
        The number of classes (possible labels).

    Returns
    -------
    np.ndarray
        A binary matrix representation of `labels` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels)
    unique = np.unique(labels)
    unique.sort()
    indexes = [np.where(unique == value)[0] for value in labels]
    if nb_classes is None:
        nb_classes = len(unique)
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(indexes)] = 1
    return categorical


class AttackDataset:
    """
    Class for splitting data into training and test sets for membership and attribute inference attacks.

    Parameters
    ----------
    x : np.ndarray or tuple
        Input data. If tuple, the first element is the training data and the second element is the test data.
    y : np.ndarray or tuple, optional
        Labels. If tuple, the first element is the training labels and the second element is the test labels.
    attack_train_ratio : float, optional
        The ratio of training data to use for the attack. Default is 0.5.
    """

    def __init__(self, x, y=None, attack_train_ratio: Optional[float] = 0.5):
        if type(x) is tuple:
            self.x_train, self.x_test = x
            self.attack_train_size = int(len(self.x_train) * attack_train_ratio)
            self.attack_test_size = int(len(self.x_test) * attack_train_ratio)
        else:
            self.x_train = x
            self.attack_train_size = int(len(self.x_train) * attack_train_ratio)

        self.y_output = y is not None

        if self.y_output:
            if type(y) is tuple:
                self.y_train, self.y_test = y
            else:
                self.y_train = y

    def membership_inference_train(self):
        """
        Get the training set for the membership inference attack.

        Returns
        -------
        tuple
            Tuple containing the training data and the membership
        """
        x = np.concatenate([self.x_train[: self.attack_train_size :], self.x_test[: self.attack_test_size]])
        train_membership = np.ones(self.attack_train_size)
        test_membership = np.zeros(self.attack_test_size)
        membership = np.concatenate([train_membership, test_membership])

        if not self.y_output:
            return x, membership

        y = np.concatenate([self.y_train[: self.attack_train_size :], self.y_test[: self.attack_test_size]])
        return x, y, membership

    def membership_inference_test(self):
        """
        Get the test set for the membership inference attack.

        Returns
        -------
        tuple
            Tuple containing the test data and the membership
        """
        x = np.concatenate([self.x_train[self.attack_train_size :], self.x_test[self.attack_test_size :]])
        train_membership = np.ones(self.attack_train_size)
        test_membership = np.zeros(self.attack_test_size)
        membership = np.concatenate([train_membership, test_membership])

        if not self.y_output:
            return x, membership

        y = np.concatenate([self.y_train[self.attack_train_size :], self.y_test[self.attack_test_size :]])
        return x, y, membership

    def attribute_inference_train(self):
        """
        Get the training set for the attribute inference attack.

        Returns
        -------
        tuple
            Tuple containing the training data and the attribute.
        """
        attack_x_train = self.x_train[: self.attack_train_size]

        if not self.y_output:
            return attack_x_train

        attack_y_train = self.y_train[: self.attack_train_size]
        return attack_x_train, attack_y_train

    def attribute_inference_test(self):
        """
        Get the test set for the attribute inference attack.

        Returns
        -------
        tuple
            Tuple containing the test data and the attribute.
        """
        attack_x_test = self.x_train[self.attack_train_size :]

        if not self.y_output:
            return attack_x_test

        attack_y_test = self.y_train[self.attack_train_size :]
        return attack_x_test, attack_y_test


class AttributeInferenceDataPreprocessor:
    """
    Class for preprocessing data for attribute inference attacks.

    Parameters
    ----------
    attack_feature : int or slice
        The feature to be attacked.
    is_regression : bool, optional
        Whether the attack is a regression attack. Default is False.
    scale_range : tuple, optional
        The range to scale the labels to. Default is None.
    prediction_normal_factor : float, optional
        The factor to normalize the predictions by. Default is 1.
    """

    def __init__(self, attack_feature, is_regression=None, scale_range=None, prediction_normal_factor=None):
        self.is_regression = is_regression  # if RegressorMixin in type(self.estimator).__mro__:
        self.scale_range = scale_range
        self.prediction_normal_factor = prediction_normal_factor
        self.attack_feature = attack_feature

    def fit_transform(self, x, y=None, pred=None):
        """
        Prepare the data for training the attack model.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        y : np.ndarray, optional
            The true labels.

        Returns
        -------
        np.ndarray or tuple
            The training data for the attack model. If `y` is provided, a tuple is returned with the training data and
            the labels.
        """
        y_ready = self._get_feature_labels(x)
        x_ready = np.delete(x, self.attack_feature, 1)

        # create training set for attack model
        if y is not None:
            normalized_labels = self._normalized_labels(y)
            x_ready = np.c_[x_ready, normalized_labels].astype(np.float32)

        if pred is not None:
            normalized_labels = self._normalized_labels(pred)
            x_ready = np.c_[x_ready, normalized_labels].astype(np.float32)

        if y_ready is None:
            return x_ready
        return x_ready, y_ready

    def transform(self, x, y=None, pred=None):
        """
        Prepare the data for inference with the attack model.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        y : np.ndarray, optional
            The true labels.

        Returns
        -------
        np.ndarray or tuple
            The data for the attack model. If `y` is provided, a tuple is returned with the data and the labels.
        """
        x_ready = x  # np.delete(x, self.attack_feature, 1)

        # create training set for attack model
        if y is not None:
            normalized_labels = self._normalized_labels(y)
            x_ready = np.c_[x_ready, normalized_labels].astype(np.float32)

        if pred is not None:
            normalized_labels = self._normalized_labels(pred)
            x_ready = np.c_[x_ready, normalized_labels].astype(np.float32)

        return x_ready

    def _get_feature_labels(self, x):
        attacked_feature = x[:, self.attack_feature]

        self._values = get_feature_values(attacked_feature, isinstance(self.attack_feature, int))
        self._nb_classes = len(self._values)

        if isinstance(self.attack_feature, int):
            y_one_hot = float_to_categorical(attacked_feature)
        else:
            y_one_hot = floats_to_one_hot(attacked_feature)

        y_ready = check_and_transform_label_format(y_one_hot, nb_classes=self._nb_classes, return_one_hot=True)
        return y_ready

    def _normalized_labels(self, y):
        if self.is_regression:
            if self.scale_range is not None:
                normalized_labels = minmax_scale(y, feature_range=self.scale_range)
            else:
                normalized_labels = y * self.prediction_normal_factor
            normalized_labels = normalized_labels.reshape(-1, 1)
        else:
            normalized_labels = check_and_transform_label_format(y, nb_classes=None, return_one_hot=True)
        return normalized_labels
