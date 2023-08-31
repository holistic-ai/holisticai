"""
The :mod:`holisticai.explanability.metrics.plots` module includes bias plotters.
"""
# Global Plots
from ._bar import bar
from ._lolipop import lolipop

# All explainability plotters
__all__ = ["bar", "lolipop"]
