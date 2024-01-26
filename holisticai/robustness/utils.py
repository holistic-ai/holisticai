import numpy as np


def load_diabetes(raw: bool = False, test_set: float = 0.3):
    import numpy as np

    """
    Loads the Diabetes Regression dataset from sklearn.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :param test_set: Proportion of the data to use as validation split. The value should be between 0 and 1.
    :return: Entire dataset and labels.
    """
    from sklearn.datasets import load_diabetes as load_diabetes_sk

    diabetes = load_diabetes_sk()
    data = diabetes.data
    if not raw:
        data /= np.amax(data, axis=0)
    targets = diabetes.target

    min_, max_ = np.amin(data), np.amax(data)

    # Shuffle data set
    random_indices = np.random.permutation(len(data))
    data, targets = data[random_indices], targets[random_indices]

    # Split training and test sets
    split_index = int((1 - test_set) * len(data))
    x_train = data[:split_index]
    y_train = targets[:split_index]
    x_test = data[split_index:]
    y_test = targets[split_index:]

    return (x_train, y_train), (x_test, y_test), min_, max_


def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1

    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = (
            score / num_positive_predicted
        )  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = (
            score / num_positive_actual
        )  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall


from holisticai.pipeline import Pipeline


def is_instance_valid(estimator, instances_allowed) -> bool:
    """
    Checks if the given estimator satisfies the requirements for this attack.

    :param estimator: The estimator to check.
    :param instances_allowed: all type of instances allowed.
    :return: True if the estimator is valid.
    """
    if is_pipeline(estimator):
        model = model_in_pipeline(estimator)
        return is_instance_valid(model, instances_allowed)

    if not (type(instances_allowed) is list):
        instances_allowed = [instances_allowed]

    for instance_type in instances_allowed:
        if isinstance(estimator, instance_type):
            return True
    return False


def is_estimator_valid(estimator, estimator_requirements) -> bool:
    """
    Checks if the given estimator satisfies the requirements for this attack.

    :param estimator: The estimator to check.
    :param estimator_requirements: Estimator requirements.
    :return: True if the estimator is valid for the attack.
    """

    if is_pipeline(estimator):
        model = model_in_pipeline(estimator)
        return is_estimator_valid(model, estimator_requirements)

    for req in estimator_requirements:
        # A requirement is either a class which the estimator must inherit from, or a tuple of classes and the
        # estimator is required to inherit from at least one of the classes
        if isinstance(req, tuple):
            if all(p not in type(estimator).__mro__ for p in req):
                return False
        elif req not in type(estimator).__mro__:
            return False
    return True


def is_pipeline(estimator):
    if type(estimator) is Pipeline:
        return True

    # if hasattr(estimator, 'model'):
    #    if type(estimator.model) is Pipeline:
    #        return True

    return False


def model_in_pipeline(estimator):
    return estimator.estimator_hdl.estimator.obj


def hai_isinstance(estimator, estimator_type):
    if is_pipeline(estimator):
        model = model_in_pipeline(estimator)
        return hai_isinstance(model, estimator_type)
    else:
        return isinstance(estimator, estimator_type)


def train_sklearn_classifier(x_train, y_train):
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)

    return model


def train_sklearn_regressor(x_train, y_train):
    from sklearn.tree import DecisionTreeRegressor

    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)

    return model


def train_holisticai_regressor(x_train, y_train):
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    from holisticai.pipeline import Pipeline

    model = Pipeline(
        steps=[("scaler", StandardScaler()), ("model", DecisionTreeRegressor())]
    )
    model.fit(x_train, y_train)

    return model


from typing import Optional, Tuple, Union

import numpy as np
from sklearn.preprocessing import minmax_scale

from holisticai.robustness.mitigation.utils.formatting import (
    check_and_transform_label_format,
    float_to_categorical,
    floats_to_one_hot,
    get_feature_index,
    get_feature_values,
)


class AttackDataset:
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

        x = np.concatenate(
            [
                self.x_train[: self.attack_train_size :],
                self.x_test[: self.attack_test_size],
            ]
        )
        train_membership = np.ones(self.attack_train_size)
        test_membership = np.zeros(self.attack_test_size)
        membership = np.concatenate([train_membership, test_membership])

        if not self.y_output:
            return x, membership

        y = np.concatenate(
            [
                self.y_train[: self.attack_train_size :],
                self.y_test[: self.attack_test_size],
            ]
        )
        return x, y, membership

    def membership_inference_test(self):
        x = np.concatenate(
            [
                self.x_train[self.attack_train_size :],
                self.x_test[self.attack_test_size :],
            ]
        )
        train_membership = np.ones(self.attack_train_size)
        test_membership = np.zeros(self.attack_test_size)
        membership = np.concatenate([train_membership, test_membership])

        if not self.y_output:
            return x, membership

        y = np.concatenate(
            [
                self.y_train[self.attack_train_size :],
                self.y_test[self.attack_test_size :],
            ]
        )
        return x, y, membership

    def attribute_inference_train(self):
        attack_x_train = self.x_train[: self.attack_train_size]

        if not self.y_output:
            return attack_x_train

        attack_y_train = self.y_train[: self.attack_train_size]
        return attack_x_train, attack_y_train

    def attribute_inference_test(self):
        attack_x_test = self.x_train[self.attack_train_size :]

        if not self.y_output:
            return attack_x_test

        attack_y_test = self.y_train[self.attack_train_size :]
        return attack_x_test, attack_y_test


class AttributeInferenceDataPreprocessor:
    def __init__(
        self,
        attack_feature,
        is_regression=None,
        scale_range=None,
        prediction_normal_factor=None,
    ):
        self.is_regression = (
            is_regression  # if RegressorMixin in type(self.estimator).__mro__:
        )
        self.scale_range = scale_range
        self.prediction_normal_factor = prediction_normal_factor
        self.attack_feature = attack_feature

    def fit_transform(self, x, y=None, pred=None):
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
        else:
            return x_ready, y_ready

    def transform(self, x, y=None, pred=None):

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

        self._values = get_feature_values(
            attacked_feature, isinstance(self.attack_feature, int)
        )
        self._nb_classes = len(self._values)

        if isinstance(self.attack_feature, int):
            y_one_hot = float_to_categorical(attacked_feature)
        else:
            y_one_hot = floats_to_one_hot(attacked_feature)

        y_ready = check_and_transform_label_format(
            y_one_hot, nb_classes=self._nb_classes, return_one_hot=True
        )
        return y_ready

    def _normalized_labels(self, y):
        if self.is_regression:
            if self.scale_range is not None:
                normalized_labels = minmax_scale(y, feature_range=self.scale_range)
            else:
                normalized_labels = y * self.prediction_normal_factor
            normalized_labels = normalized_labels.reshape(-1, 1)
        else:
            normalized_labels = check_and_transform_label_format(
                y, nb_classes=None, return_one_hot=True
            )
        return normalized_labels


import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _load_diabetes(return_X_y=False, as_frame=True):
    import pandas as pd
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    return_X_y = False
    as_frame = True

    dataset = fetch_openml(
        name="Diabetes(scikit-learn)",
        return_X_y=return_X_y,
        as_frame=as_frame,
    )
    target, labels = pd.factorize(dataset["target"])
    labels = list(labels)
    target = pd.DataFrame(target)

    df = pd.concat([dataset["data"], target], axis=1)

    df_clean = df.iloc[
        :, [i for i, n in enumerate(df.isna().sum(axis=0).T.values) if n < 1000]
    ]
    df_clean = df_clean.dropna()

    scalar = StandardScaler()
    df_t = scalar.fit_transform(df_clean)
    X = df_t[:, :-1]
    y = df_t[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=True
    )
    train_data = X_train, y_train
    test_data = X_test, y_test

    return train_data, test_data


def load_diabetes(raw: bool = False, test_set: float = 0.3):
    """
    Loads the Diabetes Regression dataset from sklearn.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :param test_set: Proportion of the data to use as validation split. The value should be between 0 and 1.
    :return: Entire dataset and labels.
    """
    from sklearn.datasets import load_diabetes as load_diabetes_sk

    diabetes = load_diabetes_sk()
    data = diabetes.data
    if not raw:
        data /= np.amax(data, axis=0)
    targets = diabetes.target

    min_, max_ = np.amin(data), np.amax(data)

    # Shuffle data set
    random_indices = np.random.permutation(len(data))
    data, targets = data[random_indices], targets[random_indices]

    # Split training and test sets
    split_index = int((1 - test_set) * len(data))
    x_train = data[:split_index]
    y_train = targets[:split_index]
    x_test = data[split_index:]
    y_test = targets[split_index:]

    return (x_train, y_train), (x_test, y_test)


import openml


def load_nursery(
    raw: bool = False,
    scaled: bool = True,
    test_set: float = 0.2,
    transform_social: bool = False,
):
    """
    Loads the UCI Nursery dataset from `config.ART_DATA_PATH` or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, categorical data is one-hot
                encoded and data is scaled using sklearn's StandardScaler according to the value of `scaled`.
    :param scaled: `True` if data should be scaled.
    :param test_set: Proportion of the data to use as validation split. The value should be between 0 and 1.
    :param transform_social: If `True`, transforms the social feature to be binary for the purpose of attribute
                             inference. This is done by assigning the original value 'problematic' the new value 1, and
                             the other original values are assigned the new value 0.
    :return: Entire dataset and labels as numpy array.
    """
    import pandas as pd
    import sklearn.preprocessing

    # # Download data if needed
    # path = "holisticai/robustness/nursery.data"
    # data = pd.read_csv(path, sep=",", names=features_names, engine="python")
    #
    # # load data
    categorical_features = [
        "parents",
        "has_nurs",
        "form",
        "housing",
        "finance",
        "social",
        "health",
    ]
    nursery = openml.datasets.get_dataset(dataset_id="nursery", version=1)
    data, *_ = nursery.get_data(dataset_format="dataframe")
    # Rename the columns as per the original dataset's feature names
    data.columns = [
        "parents",
        "has_nurs",
        "form",
        "children",
        "housing",
        "finance",
        "social",
        "health",
        "label",
    ]
    # remove rows with missing label or too sparse label
    data = data.dropna(subset=["label"])
    data.drop(data.loc[data["label"] == "recommend"].index, axis=0, inplace=True)

    if data["children"].dtype.name == "category":
        if 0 not in data["children"].cat.categories:
            data["children"] = data["children"].cat.add_categories([0])
    data["children"] = data["children"].fillna(0)

    for col in [
        "parents",
        "has_nurs",
        "form",
        "housing",
        "finance",
        "social",
        "health",
    ]:
        if "other" not in data[col].cat.categories:
            data[col] = data[col].cat.add_categories("other")
    data[col] = data[col].fillna("other")

    def modify_label(value):
        label_map = {
            "not_recom": 0,
            "very_recom": 1,
            "priority": 2,
            "spec_prior": 3,
            "recommend": 4,  # Handle the "recommend" label.
        }
        # Use label_map to get the label value, default to -1 or another sentinel value for unexpected labels
        return label_map.get(value, -1)

    data["label"] = data["label"].apply(modify_label)

    data["children"] = data["children"].apply(lambda x: 4 if x == "more" else x)

    if transform_social:

        def modify_social(value):
            if value == "problematic":
                return 1
            return 0

        data["social"] = data["social"].apply(modify_social)
        categorical_features.remove("social")

    if not raw:
        # one-hot-encode categorical features
        features_to_remove = []
        for feature in categorical_features:
            all_values = data.loc[:, feature]
            values = list(all_values.unique())
            data[feature] = pd.Categorical(
                data.loc[:, feature], categories=values, ordered=False
            )
            one_hot_vector = pd.get_dummies(data[feature], prefix=feature)
            data = pd.concat([data, one_hot_vector], axis=1)
            features_to_remove.append(feature)
        data = data.drop(features_to_remove, axis=1)

        # normalize data
        if scaled:
            label = data.loc[:, "label"]
            features = data.drop(["label"], axis=1)
            scaler = sklearn.preprocessing.StandardScaler()
            scaler.fit(features)
            scaled_features = pd.DataFrame(
                scaler.transform(features), columns=features.columns
            )
            data = pd.concat([label, scaled_features], axis=1, join="inner")

    features = data.drop(["label"], axis=1)
    if raw:
        numeric_features = (
            features.drop(categorical_features, axis=1).to_numpy().astype(np.int32)
        )
        min_, max_ = np.amin(numeric_features), np.amax(numeric_features)
    else:
        min_, max_ = np.amin(features.to_numpy().astype(np.float64)), np.amax(
            features.to_numpy().astype(np.float64)
        )

    from sklearn.model_selection import StratifiedShuffleSplit

    # Split training and test sets
    stratified = StratifiedShuffleSplit(n_splits=1, test_size=test_set, random_state=18)
    for train_set_i, test_set_i in stratified.split(data, data["label"]):
        train = data.iloc[train_set_i]
        test = data.iloc[test_set_i]
    x_train = train.drop(["label"], axis=1).to_numpy()
    y_train = train.loc[:, "label"].to_numpy()
    x_test = test.drop(["label"], axis=1).to_numpy()
    y_test = test.loc[:, "label"].to_numpy()

    if not raw and not scaled:
        x_train = x_train.astype(np.float64)
        x_test = x_test.astype(np.float64)

    return (x_train, y_train), (x_test, y_test)
