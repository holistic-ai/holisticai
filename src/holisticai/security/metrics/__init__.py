"""
The :mod:holisticai.privacy.metrics module includes attacks and privacy metrics
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from holisticai.security.metrics._anonymization import k_anonymity, l_diversity
from holisticai.security.metrics._attribute_attack import (
    AttributeAttackAccuracyScore,
    attribute_attack_accuracy_score,
)
from holisticai.security.metrics._data_minimization import (
    DataMinimizationAccuracyRatio,
    DataMinimizationMSERatio,
    data_minimization_accuracy_ratio,
    data_minimization_mse_ratio,
)
from holisticai.security.metrics._shapr import ShaprScore, shapr_score


def classification_privacy_metrics(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred_train: pd.Series,
    y_pred_test: pd.Series,
    y_pred_test_dm: dict[str, : pd.Series],
    attribute_attack: Union[str, list[str]],
):
    shapr_score = ShaprScore()
    attribute_attack_score = AttributeAttackAccuracyScore()
    dm_accuracy_ratio = DataMinimizationAccuracyRatio()

    results = []
    value = shapr_score(y_train, y_test, y_pred_train, y_pred_test)
    results.append({"metric": shapr_score.name, "value": value, "reference": shapr_score.reference})

    value = dm_accuracy_ratio(y_test, y_pred_test, y_pred_test_dm)
    results.append({"metric": dm_accuracy_ratio.name, "value": value, "reference": dm_accuracy_ratio.reference})

    value = attribute_attack_score(
        x_train, x_test, y_train, y_test, learning_task="binary_classification", attribute_attack=attribute_attack
    )
    results.append(
        {"metric": attribute_attack_score.name, "value": value, "reference": attribute_attack_score.reference}
    )

    return pd.DataFrame(results)


def regression_privacy_metrics(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred_test: pd.Series,
    y_pred_test_dm: dict[str, : pd.Series],
    attribute_attack: Union[str, list[str]],
):
    attribute_attack_score = AttributeAttackAccuracyScore()
    dm_mse_ratio = DataMinimizationMSERatio()

    results = []
    value = dm_mse_ratio(y_test, y_pred_test, y_pred_test_dm)
    results.append({"metric": dm_mse_ratio.name, "value": value, "reference": dm_mse_ratio.reference})

    value = attribute_attack_score(
        x_train, x_test, y_train, y_test, learning_task="regression", attribute_attack=attribute_attack
    )
    results.append(
        {"metric": attribute_attack_score.name, "value": value, "reference": attribute_attack_score.reference}
    )

    return pd.DataFrame(results)


__all__ = [
    "k_anonymity",
    "l_diversity",
    "classification_privacy_metrics",
    "shapr_score",
    "ShaprScore",
    "data_minimization_accuracy_ratio",
    "DataMinimizationAccuracyRatio",
    "data_minimization_mse_ratio",
    "attribute_attack_accuracy_score",
    "AttributeAttackAccuracyScore",
]
