# Imports
import numpy as np
import pandas as pd
from numpy.testing import assert_approx_equal

# Classification
from holisticai.bias.metrics import (
    abroca,
    accuracy_diff,
    average_odds_diff,
    cohen_d,
    disparate_impact,
    equal_opportunity_diff,
    false_negative_rate_diff,
    false_positive_rate_diff,
    four_fifths,
    statistical_parity,
    success_rate,
    true_negative_rate_diff,
    z_test_diff,
    z_test_ratio,
)

# Formatting
from holisticai.utils import extract_columns

# Dataset
from tests.bias.utils import load_bias_classification_data

df_c = load_bias_classification_data()

# Format data into numpy
group_a, group_b, y_pred, y_true = extract_columns(
    df_c, cols=["group_a", "group_b", "y_pred", "y_true"]
)


def test_statistical_parity():
    """test statistical_parity"""
    a = statistical_parity(group_a, group_b, y_pred)
    b = 5 / 12
    assert_approx_equal(a, b)


def test_disparate_impact():
    """test disparate_impact"""
    a = disparate_impact(group_a, group_b, y_pred)
    b = 9 / 4
    assert_approx_equal(a, b)


def test_success_rate():
    sr_a = (
        df_c[df_c["group_a"] == 1]["y_pred"].sum() / df_c[df_c["group_a"] == 1].shape[0]
    )
    assert success_rate(group_a, group_b, y_pred)["sr_a"] == sr_a


def test_four_fifths():
    """test four_fifths"""
    a = four_fifths(group_a, group_b, y_pred)
    b = 4 / 9
    assert_approx_equal(a, b)


def test_cohen_d():
    """test cohen_d"""
    a = cohen_d(group_a, group_b, y_pred)
    b = 0.9109750373485539
    assert_approx_equal(a, b)


def test_z_test_diff():
    """test z_test_diff"""
    a = z_test_diff(group_a, group_b, y_pred)
    b = 1.290994449
    assert_approx_equal(a, b)


def test_z_test_ratio():
    """test z_test_ratio"""
    a = z_test_ratio(group_a, group_b, y_pred)
    b = 1.256287689
    assert_approx_equal(a, b)


def test_equal_opportunity_diff():
    """test equal_oppotunity_diff"""
    a = equal_opportunity_diff(group_a, group_b, y_pred, y_true)
    b = 1 / 3
    assert_approx_equal(a, b)


def false_negative_rate_diff():
    """test false_negative_rate_diff"""
    a = false_negative_rate_diff(group_a, group_b, y_pred, y_true)
    b = -1 / 3
    assert_approx_equal(a, b)


def true_negative_rate_diff():
    """test true_negative_rate_diff"""
    a = true_negative_rate_diff(group_a, group_b, y_pred, y_true)
    b = -1 / 2
    assert_approx_equal(a, b)


def test_false_positive_rate_diff():
    """test false_positive_rate_diff"""
    a = false_positive_rate_diff(group_a, group_b, y_pred, y_true)
    b = 1 / 2
    assert_approx_equal(a, b)


def test_average_odds_diff():
    """test average_odds_diff"""
    a = average_odds_diff(group_a, group_b, y_pred, y_true)
    b = 5 / 12
    assert_approx_equal(a, b)


def test_accuracy_diff():
    """test accuracy_diff"""
    a = accuracy_diff(group_a, group_b, y_pred, y_true)
    b = 3 / 4 - 5 / 6
    assert_approx_equal(a, b)
