"""
The :mod:holisticai.privacy.metrics module includes attacks and privacy metrics
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from holisticai.security.metrics._anonymization import k_anonymity, l_diversity
from holisticai.security.metrics._attribute_attack import AttributeAttackScore, attribute_attack_score
from holisticai.security.metrics._data_minimization import (
    DataMinimizationAccuracyRatio,
    DataMinimizationMSERatio,
    data_minimization_score,
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
    dm_accuracy_ratio = DataMinimizationAccuracyRatio()
    attr_attack_score = AttributeAttackScore()

    results = []
    value = shapr_score(y_train, y_test, y_pred_train, y_pred_test)
    results.append({"metric": shapr_score.name, "value": value, "reference": shapr_score.reference})

    value = dm_accuracy_ratio(y_test, y_pred_test, y_pred_test_dm)
    results.append({"metric": dm_accuracy_ratio.name, "value": value, "reference": dm_accuracy_ratio.reference})

    value = attr_attack_score(x_train, x_test, y_train, y_test, attribute_attack=attribute_attack)
    results.append({"metric": attr_attack_score.name, "value": value, "reference": attr_attack_score.reference})

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
    attr_attack_score = AttributeAttackScore()
    dm_mse_ratio = DataMinimizationMSERatio()

    results = []
    value = dm_mse_ratio(y_test, y_pred_test, y_pred_test_dm)
    results.append({"metric": dm_mse_ratio.name, "value": value, "reference": dm_mse_ratio.reference})

    value = attr_attack_score(
        x_train, x_test, y_train, y_test, learning_task="regression", attribute_attack=attribute_attack
    )
    results.append({"metric": attr_attack_score.name, "value": value, "reference": attr_attack_score.reference})

    return pd.DataFrame(results)


__all__ = [
    "k_anonymity",
    "l_diversity",
    "classification_privacy_metrics",
    "shapr_score",
    "ShaprScore",
    "DataMinimizationAccuracyRatio",
    "data_minimization_score",
    "attribute_attack_score",
    "AttributeAttackScore",
]
