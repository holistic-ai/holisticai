from __future__ import annotations

from typing import Any, Callable, Union

import pandas as pd
from holisticai.security.commons import BlackBoxAttack
from holisticai.security.metrics._utils import check_valid_output_type
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error


def to_numerical_or_categorical(y: pd.Series):
    if y.dtype.kind in ["f"]:
        return y

    if y.dtype.kind in ["i", "u"]:
        return y.astype("category")

    if len(y.unique()) < 2:
        raise ValueError("The target variable must have more than 1 unique value")
    return y.astype("category")


class AttributeAttackScore:
    reference: float = 0
    name: str = "Attribute Attack Score"

    def __call__(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        attribute_attack: str,
        attack_train_ratio: float = 0.5,
        metric_fn: Union[str, Callable, None] = None,
        attacker_estimator: Any = None,
    ) -> float:
        check_valid_output_type(y_train)

        y_train = to_numerical_or_categorical(y_train)
        y_test = to_numerical_or_categorical(y_test)

        if attacker_estimator is None:
            has_continous_values = x_train[attribute_attack].dtype.kind in ["i", "u", "f"]
            attacker_estimator = LinearRegression() if has_continous_values else LogisticRegression()
            if metric_fn is None:
                metric_fn = mean_squared_error if has_continous_values else accuracy_score

        if isinstance(metric_fn, str):
            if metric_fn == "accuracy":
                metric_fn = accuracy_score
            if metric_fn == "f1":
                metric_fn = f1_score
            if metric_fn == "mean_squared_error":
                metric_fn = mean_squared_error
            if metric_fn == "mean_absolute_error":
                metric_fn = mean_absolute_error

        attacker = BlackBoxAttack(
            attacker_estimator=attacker_estimator,
            attack_feature=attribute_attack,
            attack_train_ratio=attack_train_ratio,
        )

        attacker.fit(x_train, y_train)

        y_attack, y_pred_attack = attacker.transform(x_test, y_test)

        return metric_fn(y_attack, y_pred_attack)


def attribute_attack_score(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    attribute_attack: str,
    attack_train_ratio: float = 0.5,
    **kargs,
) -> float:
    """
    Calculate the accuracy score for black box attribute attack. It is done as follows:
    - The attack attribute is removed from the training data.
    - The label is added as an input feature, and a machine learning model is trained.
    - The model is used to predict the removed attribute, and the prediction is compared with the actual value.

    Parameters
    ----------
    x_train: pd.DataFrame
        The training features.
    x_test: pd.DataFrame
        The testing features.
    y_train: pd.Series
        The training labels.
    y_test: pd.Series
        The testing labels.
    attribute_attack: str
        The attribute column in the x_train dataframe to attack.
    attack_train_ratio: float
        The ratio of the attack data to the training data.

    kargs: aditional attributes are passed to AttributeAttackScore class
    Returns
    -------
        float: The accuracy score for black box attribute attack.
    """

    bb = AttributeAttackScore()
    return bb(x_train, x_test, y_train, y_test, attribute_attack, attack_train_ratio, **kargs)
