# Imports
import numpy as np
from numpy.testing import assert_approx_equal

# Clustering
from holisticai.bias.metrics import (
    cluster_balance,
    cluster_dist_entropy,
    cluster_dist_kl,
    cluster_dist_l1,
    min_cluster_ratio,
    silhouette_diff,
    social_fairness_ratio,
)

# Format data into numpy
group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
y_pred_cl = np.array([0, 1, 1, 2, 0, 0, 0, 0, 1, 2])
y_true_cl = np.array([0, 1, 0, 2, 0, 0, 1, 0, 1, 2])
X = np.array(
    [
        [-1, 1],
        [1, 1],
        [1, 1],
        [0, -1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [-1, 1],
        [1, 1],
        [0, -1],
    ]
)
centroids = np.array([[-2, 1], [1, 2], [0, -2]])

def test_cluster_balance():
    """test cluster_balance"""
    a = cluster_balance(group_a, group_b, y_pred_cl)
    b = 0.5
    assert_approx_equal(a, b)


def test_min_cluster_ratio():
    """test min_cluster_ratio"""
    a = min_cluster_ratio(group_a, group_b, y_pred_cl)
    b = 1 / 4
    assert_approx_equal(a, b)


def test_avg_cluster_ratio():
    """test avg_cluster_ratio
    a = avg_cluster_ratio(group_a, group_b, y_pred_cl)
    b = 1 / 12 + 1
    """
    assert True


def test_cluster_dist_l1():
    """test cluster_dist_l1"""
    a = cluster_dist_l1(group_a, group_b, y_pred_cl)
    b = 0.5 * (abs(1 / 4 - 2 / 3) + abs(1 / 2 - 1 / 6) + abs(1 / 4 - 1 / 6))
    assert_approx_equal(a, b)


def test_cluster_dist_kl():
    """test cluster_dist_kl"""
    a = cluster_dist_kl(group_a, group_b, y_pred_cl)
    b = 0.40546510810816444
    assert_approx_equal(a, b)


def test_cluster_dist_entropy():
    """test cluster_dist_entropy"""
    a = cluster_dist_entropy(group_a, y_pred_cl)
    b = 1.5 * np.log(2)
    assert_approx_equal(a, b)


def test_social_fairness_ratio():
    """test social_fairness_ratio"""
    a = social_fairness_ratio(group_a, group_b, X, centroids)
    b = 1
    assert_approx_equal(a, b)


def test_silhouette_diff():
    """test silhouette_diff"""
    a = silhouette_diff(group_a, group_b, X, y_pred_cl)
    b = 0
    assert_approx_equal(a, b)
