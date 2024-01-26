from typing import List, Optional, Union

import numpy as np


class AttributeInferenceBase:
    def _feature_encoding(self, y_attack):
        self.values_, y_attack = np.unique(y_attack, return_inverse=True)
        y_attack_ready = one_hot(y_attack)
        return y_attack_ready

    def _feature_decoding(self, feature_pred):
        return np.apply_along_axis(
            lambda x: self.values_[x[0]], 1, feature_pred.reshape(-1, 1)
        )


def one_hot(data: np.ndarray, nb_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`.
    :param nb_classes: The number of classes (possible labels).
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    if nb_classes is None:
        nb_classes = data.max() + 1

    shape = (data.size, nb_classes)
    one_hot = np.zeros(shape)
    rows = np.arange(data.size)
    one_hot[rows, data] = 1
    return one_hot


def get_attack_model(attack_model_type):
    if attack_model_type == "MLPClassifier":
        from sklearn.neural_network import MLPClassifier

        attack_model = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size="auto",
            learning_rate="constant",
            learning_rate_init=0.001,
            power_t=0.5,
            max_iter=2000,
            shuffle=True,
            random_state=None,
            tol=0.0001,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=False,
            validation_fraction=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            n_iter_no_change=10,
            max_fun=15000,
        )
    elif attack_model_type == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier

        attack_model = RandomForestClassifier()

    else:
        raise ValueError("Illegal value for parameter `attack_model_type`.")

    return attack_model
