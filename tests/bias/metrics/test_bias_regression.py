# Imports
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_approx_equal

# Regression
from holisticai.bias.metrics import (
    avg_score_diff,
    avg_score_ratio,
    correlation_diff,
    disparate_impact_regression,
    jain_index,
    mae_ratio,
    max_statistical_parity,
    no_disparate_impact_level,
    rmse_ratio,
    statistical_parity_auc,
    statistical_parity_regression,
    success_rate_regression,
    zscore_diff,
)

# Formatting
from holisticai.utils import extract_columns
from tests.bias.utils import load_bias_regression_data

                        # Dataset
df_r = load_bias_regression_data()

# Format data into numpy
group_a, group_b, y_pred_r, y_true_r = extract_columns(
    df_r, cols=["group_a", "group_b", "y_pred", "y_true"]
)

def test_success_rate_regression():
    y_pred = [20, 30, 12, 45]
    group_a = [1, 1, 0, 0]
    group_b = [0, 0, 1, 1]
    assert (
        success_rate_regression(group_a, group_b, y_pred, threshold=21)["sr_a"] == 0.5
    )
    assert (
        success_rate_regression(group_a, group_b, y_pred, threshold="median")["sr_b"]
        == 0.5
    )
    with pytest.raises(ValueError) as e_info:
        success_rate_regression(group_a, group_b, y_pred, threshold="typo")


def test_disparate_impact_regression():
    """test disparate_impact_regression"""
    a = disparate_impact_regression(group_a, group_b, y_pred_r, q=0.5)
    b = 1
    assert_approx_equal(a, b)


def test_statistical_parity_regression():
    """test statistical_parity_regression"""
    a = statistical_parity_regression(group_a, group_b, y_pred_r, q=0.5)
    b = 0
    assert_approx_equal(a, b)


def test_disparate_impact_regression90():
    """test disparate_impact_regression90"""
    a = disparate_impact_regression(group_a, group_b, y_pred_r, q=0.9)
    b = 0
    assert_approx_equal(a, b)


def test_disparate_impact_regression80():
    """test disparate_impact_regression80"""
    a = disparate_impact_regression(group_a, group_b, y_pred_r, q=0.8)
    b = 1.5
    assert_approx_equal(a, b)


def test_disparate_impact_regression50():
    """test disparate_impact_regression50"""
    a = disparate_impact_regression(group_a, group_b, y_pred_r, q=0.5)
    b = 1
    assert_approx_equal(a, b)


def test_no_disparate_impact_level():
    """test no_disparate_impact_level"""
    a = no_disparate_impact_level(group_a, group_b, y_pred_r)
    b = 0.7
    assert_approx_equal(a, b)


def test_avg_score_diff():
    """test avg_score_diff"""
    a = avg_score_diff(group_a, group_b, y_pred_r)
    b = 0.1 / 12
    assert_approx_equal(a, b)


def test_avg_score_ratio():
    """test avg_score_ratio"""
    a = avg_score_ratio(group_a, group_b, y_pred_r)
    b = 1.016129032
    assert_approx_equal(a, b)


def test_avg_score_diffQ80():
    """test avg_score_diffQ80"""
    a = avg_score_diff(group_a, group_b, y_pred_r, q=0.8)
    b = -0.1
    assert_approx_equal(a, b)


def test_zscore_diff():
    """test zscore_diff"""
    a = zscore_diff(group_a, group_b, y_pred_r)
    b = 0.025042098285188583
    assert_approx_equal(a, b)


def test_zscore_diffQ80():
    """test zscore_diffQ80"""
    z = np.zeros(40)
    group_a_new = np.concatenate((group_a, z))
    group_b_new = np.concatenate((group_b, z))
    y_pred_new = np.concatenate((y_pred_r, z))
    a = zscore_diff(group_a_new, group_b_new, y_pred_new, q=0.8)
    b = 0.025042098285188583
    assert_approx_equal(a, b)


def test_statistical_parity_auc():
    """test statistical_parity_auc"""
    # test 1
    group_a = np.array([1] * 100 + [0] * 100)
    group_b = np.array([0] * 100 + [1] * 100)
    y_pred = np.array(list(2 * np.linspace(0, 1, 100)) + list(np.linspace(0, 1, 100)))
    a = statistical_parity_auc(group_a, group_b, y_pred)
    b = 0.24753333333333336
    assert_approx_equal(a, b)
    # test 2
    y_pred = np.array(list(np.linspace(0, 1, 100)) + list(2 * np.linspace(0, 1, 100)))
    a = statistical_parity_auc(group_a, group_b, y_pred)
    b = 0.24753333333333336
    assert_approx_equal(a, b)


def test_max_statistical_parity():
    """test max_statistical_parity"""
    # test 1
    group_a = np.array([1] * 100 + [0] * 100)
    group_b = np.array([0] * 100 + [1] * 100)
    y_pred = np.array(list(2 * np.linspace(0, 1, 100)) + list(np.linspace(0, 1, 100)))
    a = max_statistical_parity(group_a, group_b, y_pred)
    b = 0.5
    assert_approx_equal(a, b)


def test_correlation_diff():
    """test correlation_diff"""
    a = correlation_diff(group_a, group_b, y_pred_r, y_true_r)
    b = -0.022
    assert_approx_equal(a, b, 3)


def test_correlation_diffQ80():
    """test correlation_diffQ80"""
    z = np.zeros(40)
    group_a_new = np.concatenate((group_a, z))
    group_b_new = np.concatenate((group_b, z))
    y_pred_new = np.concatenate((y_pred_r, z))
    y_true_new = np.concatenate((y_true_r, z))
    a = correlation_diff(group_a_new, group_b_new, y_pred_new, y_true_new, q=0.8)
    b = -0.022
    assert_approx_equal(a, b, 3)


def test_rmse_ratio():
    """test rmse_ratio"""
    a = rmse_ratio(group_a, group_b, y_pred_r, y_true_r)
    b = 1.3228756555322954
    assert_approx_equal(a, b)


def test_rmse_ratioQ80():
    """test rmse_ratioQ80"""
    z = np.zeros(40)
    group_a_new = np.concatenate((group_a, z))
    group_b_new = np.concatenate((group_b, z))
    y_pred_new = np.concatenate((y_pred_r, z))
    y_true_new = np.concatenate((y_true_r, z))
    a = rmse_ratio(group_a_new, group_b_new, y_pred_new, y_true_new, q=0.8)
    b = 1.3228756555322954
    assert_approx_equal(a, b)


def test_mae_ratio():
    """test mae_ratio"""
    a = mae_ratio(group_a, group_b, y_pred_r, y_true_r)
    b = 1.25
    assert_approx_equal(a, b)


def test_mae_ratioQ80():
    """test mae_ratioQ80"""
    z = np.zeros(40)
    group_a_new = np.concatenate((group_a, z))
    group_b_new = np.concatenate((group_b, z))
    y_pred_new = np.concatenate((y_pred_r, z))
    y_true_new = np.concatenate((y_true_r, z))
    a = mae_ratio(group_a_new, group_b_new, y_pred_new, y_true_new, q=0.8)
    b = 1.25
    assert_approx_equal(a, b)


def test_jain_index():
    """test jain_index"""
    a = jain_index(y_pred_r, y_true_r)
    b = 0.9307692
    assert_approx_equal(a, b)
