import os
import sys

sys.path.append(os.getcwd())
import numpy as np
import pytest
from utils import (
    numeric_config,
    numeric_nan_config,
    str_config,
    unconventional_numeric_config,
)

from holisticai.bias.metrics import confusion_matrix

##################################################################################################################
# 2 inputs
##################################################################################################################
# String and unconventional y values

metrics_functions = [confusion_matrix]


@pytest.mark.parametrize("metric_function", metrics_functions)
def test_string_y_values(metric_function):
    """metric must support string labels"""
    _, y_pred, y_true = str_config()
    metric_function(y_pred, y_true)


@pytest.mark.parametrize("metric_function", metrics_functions)
def test_unconventional_y_values(metric_function):
    """metric support unconventional numeric labels"""
    _, y_pred, y_true = unconventional_numeric_config()
    metric_function(y_pred, y_true)


@pytest.mark.xfail()
@pytest.mark.parametrize("metric_function", metrics_functions)
def test_raise_invalid_y_values(metric_function):
    """metric must validate y values in classes"""
    _, y_pred, y_true = numeric_config()
    classes = [2, 5]
    metric_function(y_pred, y_true, classes=classes)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("metric_function", metrics_functions)
def test_y_pred_nan(metric_function):
    """metric must validate non nan values in y_pred"""
    _, y_pred, y_true = numeric_nan_config(y_pred_nan=True)
    metric_function(y_pred, y_true)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("metric_function", metrics_functions)
def test_y_true_nan(metric_function):
    """metric must validate non nan values in y_true"""
    _, y_pred, y_true = numeric_nan_config(y_pred_nan=False)
    metric_function(y_pred, y_true)
