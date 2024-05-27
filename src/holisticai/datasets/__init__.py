"""
The :mod:`holisticai.datasets` module includes dataloaders for quick experimentation
"""

from holisticai.datasets._dataloaders import (
    load_adult,
    load_heart,
    load_last_fm,
    load_law_school,
    load_student,
    load_us_crime,
)
from holisticai.datasets._dataset import load_dataset
from holisticai.datasets._make_classification import make_classification
from holisticai.datasets._make_multiclass import make_multiclass
from holisticai.datasets._make_regression import make_regression

__all__ = [
    "load_adult",
    "load_student",
    "load_law_school",
    "load_last_fm",
    "load_us_crime",
    "load_heart",
    "load_dataset",
    "make_classification",
    "make_multiclass",
    "make_regression",
]
