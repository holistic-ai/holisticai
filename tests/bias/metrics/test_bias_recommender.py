# Imports
import numpy as np
import pandas as pd
from numpy.testing import assert_approx_equal

# Recommender
from holisticai.bias.metrics import (
    aggregate_diversity,
    avg_f1_ratio,
    avg_precision_ratio,
    avg_recall_ratio,
    avg_recommendation_popularity,
    exposure_entropy,
    exposure_kl,
    exposure_l1,
    gini_index,
    mad_score,
    recommender_mae_ratio,
    recommender_rmse_ratio,
)

# Formatting
from holisticai.utils import extract_columns
from tests.bias.utils import load_bias_recommender_data

# Dataset
df_rec = load_bias_recommender_data()

# Format data into numpy
group_a, group_b = extract_columns(df_rec, cols=["group_a", "group_b"])
mat_pred = df_rec[["item_1", "item_2", "item_3", "item_4"]].to_numpy()
mat_true = df_rec[
    ["item_1_true", "item_2_true", "item_3_true", "item_4_true"]
].to_numpy()


def test_aggregate_diversity():
    """test aggregate_diversity"""
    a = aggregate_diversity(mat_pred, top=None, thresh=0.5)
    b = 1
    assert_approx_equal(a, b)


def test_gini_index():
    """test gini_index"""
    a = gini_index(mat_pred, top=None, thresh=0.5, normalize=True)
    b = 0.1851851851851852
    assert_approx_equal(a, b)


def test_exposure_entropy():
    """test exposure_entropy"""
    a = exposure_entropy(mat_pred, top=None, thresh=0.5, normalize=True)
    b = 1.9546859469463558 * np.log(2)
    assert_approx_equal(a, b)


def test_avg_recommendation_popularity():
    """test avg_recommendation_popularity"""
    a = avg_recommendation_popularity(mat_pred, top=None, thresh=0.5, normalize=True)
    b = 4.833333333333334
    assert_approx_equal(a, b)

    mat2 = np.concatenate((mat_pred, np.ones(mat_pred.shape) * 0.1), axis=0)
    a2 = avg_recommendation_popularity(mat2, top=None, thresh=0.5, normalize=True)
    assert_approx_equal(a2, b)


def test_mad_score():
    """test mad_score"""
    a = mad_score(group_a, group_b, mat_pred, normalize=True)
    b = 0.00925925925925919
    assert_approx_equal(a, b)


def test_exposure_l1():
    """test exposure_l1"""
    a = exposure_l1(group_a, group_b, mat_pred)
    b = 0.5 * (
        abs(2 / 7 - 3 / 12)
        + abs(2 / 7 - 4 / 12)
        + abs(1 / 7 - 2 / 12)
        + abs(2 / 7 - 3 / 12)
    )
    assert_approx_equal(a, b)


def test_exposure_kl():
    """test exposure_kl"""
    a = exposure_kl(group_a, group_a, mat_pred)
    b = 0
    assert_approx_equal(a, b)


def test_avg_precision_ratio():
    """test avg_precision_ratio"""
    group_a = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    group_b = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    a = avg_precision_ratio(group_a, group_b, mat_pred, mat_true)
    b = (1 + 2 / 3) / 2
    assert_approx_equal(a, b)


def test_avg_recall_ratio():
    """test avg_recall_ratio"""
    group_a = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    group_b = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    a = avg_recall_ratio(group_a, group_b, mat_pred, mat_true)
    b = 1
    assert_approx_equal(a, b)


def test_avg_f1_ratio():
    """test avg_f1_ratio"""
    group_a = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    group_b = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    a = avg_f1_ratio(group_a, group_b, mat_pred, mat_true)
    b = (2 * (2 / 3) / (2 / 3 + 1) + 1) / 2
    assert_approx_equal(a, b)


def test_recommender_rmse_ratio():
    """test recommender_rmse_ratio"""
    group_a = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    group_b = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    a = recommender_rmse_ratio(group_a, group_b, mat_pred, mat_true)
    b = np.sqrt(0.2**2 + 0.2**2 + 0.5**2) / np.sqrt(0.2**2)
    assert_approx_equal(a, b)


def test_recommender_mae_ratio():
    """test avg_mae_ratio"""
    group_a = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    group_b = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    a = recommender_mae_ratio(group_a, group_b, mat_pred, mat_true)
    b = 4.499999999999998
    assert_approx_equal(a, b)
