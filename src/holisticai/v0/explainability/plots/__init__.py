"""
The :mod:`holisticai.explanability.metrics.plots` module includes bias plotters.
"""
# Global Plots
from ._bar import bar
from ._importances import contrast_matrix
from ._lolipop import lolipop
from ._tree import DecisionTreeVisualizer

# All explainability plotters
__all__ = ["bar", "lolipop", "DecisionTreeVisualizer", "contrast_matrix"]
