"""
The :mod:`holisticai.bias.plots` module includes bias plotters.
"""
# Classification Plots
from ._bias_classification_plots import abroca_plot

# Exploratory plots
from ._bias_exploratory_plots import distribution_plot, group_pie_plot, histogram_plot

# Multiclass Plots
from ._bias_multiclass_plots import (
    accuracy_bar_plot,
    disparate_impact_plot,
    frequency_matrix_plot,
    frequency_plot,
    statistical_parity_plot,
)

# Recommender Plots
from ._bias_recommender_plots import (
    exposure_diff_plot,
    exposure_ratio_plot,
    long_tail_plot,
)

# Regression Plots
from ._bias_regression_plots import (
    disparate_impact_curve,
    mae_bar_plot,
    rmse_bar_plot,
    statistical_parity_curve,
    success_rate_curve,
    success_rate_curves,
)

# All bias plotters
__all__ = [
    "abroca_plot",
    "success_rate_curve",
    "disparate_impact_curve",
    "statistical_parity_curve",
    "success_rate_curves",
    "rmse_bar_plot",
    "mae_bar_plot",
    "exposure_diff_plot",
    "exposure_ratio_plot",
    "frequency_plot",
    "statistical_parity_plot",
    "disparate_impact_plot",
    "frequency_matrix_plot",
    "accuracy_bar_plot",
    "long_tail_plot",
    "group_pie_plot",
    "distribution_plot",
    "histogram_plot",
]
