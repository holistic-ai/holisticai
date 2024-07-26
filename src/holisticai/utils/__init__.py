"""
The :mod:`holisticai.utils` module includes utils helper tools
"""

# formatting
from holisticai.utils._definitions import (
    BinaryClassificationProxy,
    ConditionalImportances,
    Importances,
    LocalConditionalImportances,
    LocalImportances,
    ModelProxy,
    MultiClassificationProxy,
    PartialDependence,
    RegressionProxy,
    create_proxy,
)
from holisticai.utils._formatting import (
    extract_columns,
    extract_group_vectors,
    mat_to_binary,
    normalize_tensor,
    recommender_formatter,
)

# plotting
from holisticai.utils._plotting import get_colors
from holisticai.utils.inspection._partial_dependence import compute_partial_dependence

__all__ = [
    "extract_columns",
    "mat_to_binary",
    "normalize_tensor",
    "get_colors",
    "recommender_formatter",
    "extract_group_vectors",
    "BinaryClassificationProxy",
    "MultiClassificationProxy",
    "RegressionProxy",
    "compute_partial_dependence",
    "create_proxy",
    "Importances",
    "LocalImportances",
    "LocalConditionalImportances",
    "PartialDependence",
    "ConditionalImportances",
    "ModelProxy",
]
