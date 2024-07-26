# Imports
import numpy as np
from numpy.testing import assert_approx_equal, assert_array_almost_equal

# Multiclass
from holisticai.bias.metrics import (
    accuracy_matrix,
    confusion_tensor,
    frequency_matrix,
    multiclass_average_odds,
    multiclass_equality_of_opp,
    multiclass_statistical_parity,
    multiclass_true_rates,
    precision_matrix,
    recall_matrix,
)

# Format data into numpy for each task
g_min = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
g_maj = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

p = np.array(["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"])
y_pred_mc = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
y_true_mc = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1])

p2 = np.array(["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C"])
y_pred2 = np.array([0, 2, 1, 0, 2, 1, 0, 1, 1, 2, 0])


p_bin = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
y_pred = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0])
y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1, 1])


def test_confusion_tensor():
    """test confusion_tensor"""
    a = confusion_tensor(p, y_pred_mc, y_true_mc, as_tensor=True)
    b = np.array(
        [
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )
    assert_array_almost_equal(a, b)


def test_frequency_matrix():
    """test frequency_matrix"""
    a = frequency_matrix(p, y_pred_mc).to_numpy()
    b = np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.5, 0.0, 0.5]])
    assert_array_almost_equal(a, b)


def test_accuracy_matrix():
    """test accuracy_matrix"""
    a = accuracy_matrix(p, y_pred_mc, y_true_mc).to_numpy()
    b = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert_array_almost_equal(a, b)


def test_precision_matrix():
    """test precision_matrix"""
    a = precision_matrix(p, y_pred_mc, y_true_mc).to_numpy()
    b = np.array([[1.0, 0.5, np.nan], [0.0, 1.0, 0.0], [np.nan, 0.0, 1.0]])
    assert_array_almost_equal(a, b)


def test_recall_matrix():
    """test recall_matrix"""
    a = recall_matrix(p, y_pred_mc, y_true_mc).to_numpy()
    b = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, np.nan, 1.0]])
    assert_array_almost_equal(a, b)


def test_multiclass_equality_of_opp():
    """test multiclass equality of opportunity"""
    a = multiclass_equality_of_opp(p_bin, y_pred, y_true)
    b = 0.7
    assert_array_almost_equal(a, b)

    a = multiclass_equality_of_opp(p, y_pred_mc, y_true_mc)
    b = 0.5308628148148148 * 1.5
    assert_array_almost_equal(a, b, 5)


def test_multiclass_average_odds():
    """test multiclass average odds"""
    a = multiclass_average_odds(p_bin, y_pred, y_true)
    b = 0.3
    assert_array_almost_equal(a, b)

    a = multiclass_average_odds(p, y_pred_mc, y_true_mc)
    b = (1 / 9) * 1.5
    assert_array_almost_equal(a, b)

    a = multiclass_average_odds(p2, y_pred2, y_pred2)
    b = 0
    assert_array_almost_equal(a, b)


def test_multiclass_true_rates():
    """test multiclass true rates"""
    a = multiclass_true_rates(p_bin, y_pred, y_true)
    b = 0.7
    assert_array_almost_equal(a, b)

    a = multiclass_true_rates(p, y_pred_mc, y_true_mc)
    b = 2 / 3
    assert_array_almost_equal(a, b)

    a = multiclass_true_rates(p2, y_pred2, y_pred2)
    b = 0
    assert_array_almost_equal(a, b)


def test_multiclass_statistical_parity():
    """test multiclass statistical parity"""
    a = multiclass_statistical_parity(p_bin, y_pred)
    b = 1 / 6
    assert_array_almost_equal(a, b)

    a = multiclass_statistical_parity(p, y_pred_mc)
    b = 2 * 3 / 18
    assert_array_almost_equal(a, b)
