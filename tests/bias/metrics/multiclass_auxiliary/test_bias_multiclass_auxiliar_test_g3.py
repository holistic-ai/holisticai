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

from holisticai.bias.metrics import (
    accuracy_matrix,
    confusion_tensor,
    multiclass_average_odds,
    multiclass_equality_of_opp,
    multiclass_true_rates,
    precision_matrix,
    recall_matrix,
)

##################################################################################################################
# 3 inputs
##################################################################################################################
metric_functions = [
    multiclass_average_odds,
    multiclass_true_rates,
    multiclass_equality_of_opp,
    confusion_tensor,
    accuracy_matrix,
    precision_matrix,
    recall_matrix,
]

#######################
# String and unconventional numeric labels
@pytest.mark.parametrize("metric_function", metric_functions)
def test_support_string_y_values(metric_function):
    """metric must supoort string labels"""
    p_attr, y_pred, y_true = str_config()
    metric_function(p_attr, y_pred, y_true)


@pytest.mark.parametrize("metric_function", metric_functions)
def test_support_unvonventional_y_values(metric_function):
    """metric must support unconventional numeric labels"""
    p_attr, y_pred, y_true = unconventional_numeric_config()
    metric_function(p_attr, y_pred, y_true)


#######################
# Validation Classes and Group List
@pytest.mark.xfail()
@pytest.mark.parametrize("metric_function", metric_functions)
def test_raise_invalid_y_values(metric_function):
    """metric must validate y values in classes"""
    p_attr, y_pred, y_true = numeric_config()
    classes = [2, 5]
    metric_function(p_attr, y_pred, y_true, classes=classes)


@pytest.mark.xfail()
@pytest.mark.parametrize("metric_function", metric_functions)
def test_raise_invalid_p_attr_values_3_inputs(metric_function):
    """metric must validate p_attr values in groups"""
    p_attr, y_pred, y_true = numeric_config()
    groups = ["c", "a"]
    metric_function(p_attr, y_pred, y_true, groups=groups)


#######################
# Nan Values y_pred y_true
@pytest.mark.xfail()
@pytest.mark.parametrize("metric_function", metric_functions)
def test_y_pred_nan_3_inputs(metric_function):
    """metric must raise a ValueError for nan values"""
    p_attr, y_pred, y_true = numeric_nan_config()
    metric_function(p_attr, y_pred, y_true)


@pytest.mark.xfail()
@pytest.mark.parametrize("metric_function", metric_functions)
def test_y_true_nan_3_inputs(metric_function):
    """metric must raise a ValueError for nan values"""
    p_attr, y_pred, y_true = numeric_nan_config()
    metric_function(p_attr, y_pred, y_true)
