"""
The :mod:`holisticai.datasets` module includes dataloaders for quick experimentation
"""

from ._dataloaders import (
    load_adult,
    load_last_fm,
    load_law_school,
    load_student,
    load_us_crime,
)

__all__ = [
    "load_adult",
    "load_student",
    "load_law_school",
    "load_last_fm",
    "load_us_crime",
]
