"""
The :mod:`holisticai.datasets.plots` module includes bias plotters.
"""

from holisticai.datasets.plots._exploratory import (
    correlation_matrix_plot,
    group_confusion_matrices,
    group_pie_plot,
    histogram_plot,
)

# All bias plotters
__all__ = [
    "group_pie_plot",
    "histogram_plot",
    "correlation_matrix_plot",
    "group_confusion_matrices",
]
