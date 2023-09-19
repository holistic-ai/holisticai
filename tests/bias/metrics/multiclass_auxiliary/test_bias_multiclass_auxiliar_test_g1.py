import numpy as np
import pytest
from utils import (
    numeric_config,
    numeric_nan_config,
    str_config,
    unconventional_numeric_config,
)

from holisticai.bias.metrics import frequency_matrix, multiclass_statistical_parity

#######################
# 1 entrada + p_attr
#######################
# NaN Values
metrics_functions = [multiclass_statistical_parity, frequency_matrix]


@pytest.mark.parametrize("metric_function", metrics_functions)
def test_string_y_values(metric_function):
    """metric must accept string y values"""
    p_attr, y_pred, _ = str_config()
    metric_function(p_attr, y_pred)


@pytest.mark.parametrize("metric_function", metrics_functions)
def test_unvonventional_numeric_y(metric_function):
    """metric must accept unconventional numeric labels"""
    p_attr, y_pred, _ = unconventional_numeric_config()
    metric_function(p_attr, y_pred)


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize("metric_function", metrics_functions)
def test_y_pred_nan(metric_function):
    """metric must validate y_pred without nan values, the error must be ValueError and not KeyError"""
    p_attr, y_pred, _ = numeric_nan_config(y_pred_nan=True)
    metric_function(p_attr, y_pred)
