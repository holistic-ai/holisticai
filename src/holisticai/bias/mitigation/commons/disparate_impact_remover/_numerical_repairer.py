from __future__ import annotations

from typing import Optional

from holisticai.bias.mitigation.commons.disparate_impact_remover._categorical_repairer import CategoricalRepairer
from holisticai.bias.mitigation.commons.disparate_impact_remover._utils import (
    freedman_diaconis_bin_size as bin_calculator,
)
from holisticai.bias.mitigation.commons.disparate_impact_remover._utils import get_median, make_histogram_bins


class NumericalRepairer:
    """
    The algorithm used to repair numerical features in a dataset to mitigate disparate impact.

    Parameters
    ----------
    feature_to_repair : int
        The index of the feature to repair.
    repair_level : float
        The level of repair to apply to the feature. A value of 0 means no repair, while a value of 1 means full repair.
    kdd : bool, optional
        Whether to use the K-nearest neighbor density estimator to calculate bin sizes. Default is False.
    features_to_ignore : list, optional
        A list of feature names to ignore during repair. Default is an empty list.

    Attributes
    ----------
    feature_to_repair : int
        The index of the feature to repair.
    repair_level : float
        The level of repair to apply to the feature.
    kdd : bool
        Whether to use the K-nearest neighbor density estimator to calculate bin sizes.
    features_to_ignore : list
        A list of feature names to ignore during repair.

    Methods
    -------
    repair(data_to_repair)
        Repairs the numerical feature in the input dataset to mitigate disparate impact.
    """

    def __init__(
        self,
        feature_to_repair: int,
        repair_level: float,
        kdd: bool = False,
        features_to_ignore: Optional[list[str]] = None,
    ):
        if features_to_ignore is None:
            features_to_ignore = []
        self.feature_to_repair = feature_to_repair
        self.repair_level = repair_level
        self.kdd = kdd
        self.features_to_ignore = features_to_ignore

    def _calculate_category_medians(self, index_bins: list[list[int]], data_to_repair: list[list[float]]) -> dict:
        """
        Calculates the median value for each bin in the input dataset.

        Parameters
        ----------
        index_bins : list
            A list of indices for each bin in the dataset.
        data_to_repair : list
            The input dataset to repair.

        Returns
        -------
        dict
            A dictionary containing the median value for each bin in the dataset.
        """
        return {
            f"BIN_{i}": get_median([data_to_repair[j][self.feature_to_repair] for j in index_bin], self.kdd)
            for i, index_bin in enumerate(index_bins)
        }

    def repair(self, data_to_repair: list[list[float]]) -> list[list[float]]:
        """
        Repairs the numerical feature in the input dataset to mitigate disparate impact.

        Parameters
        ----------
        data_to_repair : list
            The input dataset to repair.

        Returns
        -------
        list
            The repaired dataset.
        """

        binned_data = [row[:] for row in data_to_repair]
        index_bins = make_histogram_bins(bin_calculator, data_to_repair, self.feature_to_repair)

        category_medians = self._calculate_category_medians(index_bins, data_to_repair)

        for i, index_bin in enumerate(index_bins):
            for j in index_bin:
                binned_data[j][self.feature_to_repair] = f"BIN_{i}"

        categoric_repairer = CategoricalRepairer(
            feature_to_repair=self.feature_to_repair,
            repair_level=self.repair_level,
            kdd=self.kdd,
            features_to_ignore=self.features_to_ignore,
        )

        repaired_data = categoric_repairer.repair(binned_data)

        for i, row in enumerate(repaired_data):
            row[self.feature_to_repair] = (
                category_medians[row[self.feature_to_repair]]
                if self.repair_level > 0
                else data_to_repair[i][self.feature_to_repair]
            )

        return repaired_data
