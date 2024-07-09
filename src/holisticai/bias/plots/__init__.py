"""
The :mod:`holisticai.bias.plots` module includes bias plotters.
"""

# Classification Plots
# Exploratory plots
# Multiclass Plots
from holisticai.bias.plots._bias_exploratory_plots import distribution_plot, group_pie_plot, histogram_plot
from holisticai.bias.plots._classification import abroca_plot
from holisticai.bias.plots._multiclass import (
    accuracy_bar_plot,
    disparate_impact_plot,
    frequency_matrix_plot,
    frequency_plot,
    statistical_parity_plot,
)

# Recommender Plots
from holisticai.bias.plots._recommender import exposure_diff_plot, exposure_ratio_plot, long_tail_plot

# Regression Plots
from holisticai.bias.plots._regression import (
    disparate_impact_curve,
    mae_bar_plot,
    rmse_bar_plot,
    statistical_parity_curve,
    success_rate_curve,
    success_rate_curves,
)

# Report Plots
from holisticai.bias.plots._report import bias_metrics_report

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
    "bias_metrics_report",
    "group_confusion_matrices",
]
