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
    load_german_credit,
    load_census_kdd,
    load_bank_marketing,
    load_compass,
    load_diabetes,
    load_acsincome,
    load_acspublic,
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
    "load_german_credit",
    "load_census_kdd",
    "load_bank_marketing",
    "load_compass",
    "load_diabetes",
    "load_acsincome",
    "load_acspublic",
]
