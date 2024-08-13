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
from holisticai.datasets._dataset import DataLoader, Dataset, DatasetDict, GroupByDataset, concatenate_datasets
from holisticai.datasets._load_dataset import load_dataset

__all__ = [
    "load_dataset",
    "Dataset",
    "DatasetDict",
    "GroupByDataset",
    "DataLoader",
    "concatenate_datasets",
    "load_adult",
    "load_last_fm",
    "load_law_school",
    "load_heart",
    "load_student",
    "load_us_crime",
]
