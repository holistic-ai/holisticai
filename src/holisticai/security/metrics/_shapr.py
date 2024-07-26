from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pandas as pd
from holisticai.utils.models.neighbors import KNeighborsClassifier
from jax.nn import one_hot
from numpy.typing import ArrayLike
from sklearn.preprocessing import LabelEncoder

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def to_categorical(labels: np.ndarray | list[float]) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`.
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=int)
    nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical


def transform_label_to_numerical_label(labels: np.ndarray, le: LabelEncoder = None) -> np.ndarray:
    return_encoder = False
    if labels.dtype.kind in {"U", "S", "O"}:  # Check if labels are strings
        if le is None:
            le = LabelEncoder()
            labels = le.fit_transform(labels)
            return_encoder = True
        else:
            labels = le.transform(labels)
            return_encoder = False
    elif le is None:
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        return_encoder = True

    if return_encoder:
        return np.array(labels), le
    return np.array(labels)


def check_and_transform_label_format(labels: np.ndarray) -> jnp.ndarray:
    """
    Check label format and transform to one-hot-encoded labels if necessary

    :param labels: An array of integer or string labels of shape `(nb_samples,)`, `(nb_samples, 1)` or `(nb_samples, nb_classes)`.
    :return: Labels with shape `(nb_samples, nb_classes)` (one-hot) or `(nb_samples,)` (index).
    """

    labels = jnp.array(labels)
    return_one_hot = True
    labels_return = labels
    binary_case = 2

    if len(labels.shape) == binary_case and labels.shape[1] > 1:  # multi-class, one-hot encoded
        if not return_one_hot:
            labels_return = jnp.argmax(labels, axis=1)
            labels_return = jnp.expand_dims(labels_return, axis=1)
    elif len(labels.shape) == binary_case and labels.shape[1] == 1:
        nb_classes = int(jnp.max(labels) + 1)
        if nb_classes > binary_case:  # multi-class, index labels
            labels_return = one_hot(labels, nb_classes) if return_one_hot else jnp.expand_dims(labels, axis=1)
        elif nb_classes == binary_case:  # binary, index labels
            if return_one_hot:
                labels_return = one_hot(labels, nb_classes)
        else:
            msg = (
                "Shape of labels not recognised."
                "Please provide labels in shape (nb_samples,) or (nb_samples, nb_classes)"
            )
            raise ValueError(msg)
    elif len(labels.shape) == 1:  # index labels
        labels_return = one_hot(labels, int(jnp.max(labels) + 1)) if return_one_hot else jnp.expand_dims(labels, axis=1)
    else:
        msg = (
            "Shape of labels not recognised." "Please provide labels in shape (nb_samples,) or (nb_samples, nb_classes)"
        )
        raise ValueError(msg)

    return labels_return


class ShaprScore:
    reference: float = 0
    name: str = "SHAPr"

    def __call__(
        self,
        y_train: ArrayLike,
        y_test: ArrayLike,
        y_pred_train: ArrayLike,
        y_pred_test: ArrayLike,
        batch_size=500,
        train_size=1.0,
    ) -> jnp.ndarray:
        y_train, le = transform_label_to_numerical_label(y_train)
        y_test = transform_label_to_numerical_label(y_test, le)
        y_pred_train = transform_label_to_numerical_label(y_pred_train, le)
        y_pred_test = transform_label_to_numerical_label(y_pred_test, le)

        y_train = check_and_transform_label_format(y_train)
        y_test = check_and_transform_label_format(y_test)

        y_pred_train = y_pred_train.reshape([-1, 1])
        y_pred_test = y_pred_test.reshape([-1, 1])

        n_train_samples = int(train_size * len(y_train))

        knn = KNeighborsClassifier()
        knn.fit(y_pred_train, y_train)
        batch_generator = knn.kneighbors_batched(
            y_pred_test, n_neighbors=n_train_samples, return_distance=False, batch_size=batch_size
        )
        n_test = y_pred_test.shape[0]

        results = []
        for i, batch in enumerate(batch_generator):
            n_indexes = jnp.array(batch)
            n_indexes = n_indexes[:, ::-1]
            sorted_indexes = jnp.argsort(n_indexes, axis=1)
            b_y_test = y_test[i * batch_size : (i + 1) * batch_size]
            sorted_y_train = y_train[n_indexes]
            y_test_rep = jnp.repeat(b_y_test[:, jnp.newaxis], n_indexes.shape[1], axis=1)
            y_indicator = jnp.all(sorted_y_train == y_test_rep, axis=2).astype(int)
            d_phi_y = (y_indicator[:, 1:] - y_indicator[:, :-1]) / (n_train_samples - jnp.arange(1, n_train_samples))
            d_phi_y = jnp.concatenate([y_indicator[:, 0:1] / n_train_samples, d_phi_y], axis=1)
            phi_y = jnp.cumsum(d_phi_y, axis=1)
            results_test_sorted = jnp.take_along_axis(phi_y, sorted_indexes, axis=1)
            results.append(results_test_sorted)

        results = jnp.concatenate(results, axis=0)
        per_sample = jnp.sum(results, axis=0)
        sum_per_sample = per_sample * n_train_samples / n_test
        return float(sum_per_sample.mean())


def shapr_score(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred_train: pd.Series,
    y_pred_test: pd.Series,
    batch_size=500,
    train_size=1.0,
) -> jnp.ndarray:
    """
    Compute the SHAPr membership privacy risk metric for the given classifier and training set.

    Parameters
    ----------

    y_train: pd.Series
        (nb_samples, nb_classes) or indices of shape (nb_samples,). Target values (class labels) of `x_train`, one-hot-encoded.

    y_test: pd.Series
        (nb_samples, nb_classes) or indices of shape (nb_samples,). Target values (class labels) of `x_test`, one-hot-encoded.

    y_pred_train: pd.Series
        (nb_samples, nb_classes) or indices of shape (nb_samples,). Predicted values (class labels) of `x_train`, one-hot-encoded.

    y_pred_test: pd.Series
        (nb_samples, nb_classes) or indices of shape (nb_samples,). Predicted values (class labels) of `x_test`, one-hot-encoded.

    batch_size: int, default=100
        The number of samples to process in each batch.

    train_size: float, default=1.0
        The fraction of the training set to use for the k-nearest neighbors search

    Returns
    -------
        float: The higher the value, the higher the privacy leakage for that sample. Any value above 0 should be considered a privacy leak.

    Reference
    ---------
    .. [1] https://arxiv.org/abs/2112.02230

    """
    shapr = ShaprScore()
    return shapr(y_train, y_test, y_pred_train, y_pred_test, batch_size, train_size)
