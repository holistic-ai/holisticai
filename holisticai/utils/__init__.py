"""
The :mod:`holisticai.utils` module includes utils helper tools
"""

# formatting
from ._formatting import (
    extract_columns,
    extract_group_vectors,
    mat_to_binary,
    normalize_tensor,
    recommender_formatter,
)

# plotting
from ._plotting import get_colors

__all__ = [
    "extract_columns",
    "mat_to_binary",
    "normalize_tensor",
    "get_colors",
    "recommender_formatter",
    "extract_group_vectors",
]
