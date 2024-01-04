"""
The :mod:`holisticai.datasets` module includes dataloaders for quick experimentation
"""

from ._dataloaders import (
    load_adult,
    load_heart,
    load_last_fm,
    load_law_school,
    load_student,
    load_us_crime,
)
from ._preprocessers import load_dataset

__all__ = [
    "load_adult",
    "load_student",
    "load_law_school",
    "load_last_fm",
    "load_us_crime",
    "load_heart",
    "load_dataset",
]
