import pandas as pd
from holisticai.security.commons import BlackBoxAttack
from sklearn.metrics import accuracy_score


def _check_categorical_labels(y: pd.Series):
    if len(y.unique()) < 2:
        raise ValueError("The target variable must have more than 1 unique value")
    y = y.astype("category")
    return y


def _check_regression_outputs(y: pd.Series):
    if y.dtype.kind not in ["i", "u", "f"]:
        raise ValueError("The target variable must contain real values for regression tasks")


class AttributeAttackAccuracyScore:
    reference: float = 0
    name: str = "Attribute Attack Accuracy Score"

    def __call__(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        learning_task: str,
        attribute_attack: str,
    ) -> float:
        if learning_task in ["binary_classification", "multi_classification"]:
            y_train = _check_categorical_labels(y_train)
            y_test = _check_categorical_labels(y_test)
        elif learning_task == "regression":
            _check_regression_outputs(y_train)
            _check_regression_outputs(y_test)
        else:
            raise ValueError(
                "The learning task must be one of 'binary_classification', 'multi_classification', or 'regression'"
            )

        attacker = BlackBoxAttack(attack_feature=attribute_attack, attack_train_ratio=0.5)

        attacker.fit(x_train, y_train)

        y_attack, y_pred_attack = attacker.transform(x_test, y_test)

        return accuracy_score(y_attack, y_pred_attack)


def attribute_attack_accuracy_score(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    learning_task: str,
    attribute_attack: str,
) -> float:
    """
    Calculate the accuracy score for black box attribute attack.

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
    learning_task: str
        The learning task.
    attribute_attack: str
        The attribute to attack.

    Returns
    -------
        float: The accuracy score for black box attribute attack.
    """
    bb = AttributeAttackAccuracyScore()
    return bb(x_train, x_test, y_train, y_test, learning_task, attribute_attack)
