"""
The :mod:`holisticai.plots.bias` module includes bias plotters.
"""
# Classification Plots
# Exploratory plots
# Multiclass Plots
from holisticai.plots.bias._bias_classification_plots import abroca_plot
from holisticai.plots.bias._bias_exploratory_plots import (  # type: ignore
    correlation_matrix_plot,
    distribution_plot,
    group_pie_plot,
    histogram_plot,
)
from holisticai.plots.bias._bias_multiclass_plots import (
    accuracy_bar_plot,
    disparate_impact_plot,
    frequency_matrix_plot,
    frequency_plot,
    statistical_parity_plot,
)

# Recommender Plots
from holisticai.plots.bias._bias_recommender_plots import exposure_diff_plot, exposure_ratio_plot, long_tail_plot

# Regression Plots
from holisticai.plots.bias._bias_regression_plots import (
    disparate_impact_curve,
    mae_bar_plot,
    rmse_bar_plot,
    statistical_parity_curve,
    success_rate_curve,
    success_rate_curves,
)

# Report Plots
from holisticai.plots.bias._bias_report_plots import bias_metrics_report

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
    "correlation_matrix_plot",
    "bias_report_classification",
    "bias_report_regression",
    "bias_report_clustering",
    "bias_metrics_report",
]
