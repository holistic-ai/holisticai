from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class GridGenerator:
    """
    Create a grid using a constraint
    """

    def __init__(
        self,
        grid_size: Optional[int] = 5,
        grid_limit: Optional[float] = 2.0,
        neg_allowed: Optional[np.ndarray] = None,
        force_L1_norm: Optional[bool] = True,
    ):
        """
        Initialize Grid Generator

        Parameters
        ----------
        grid_size: int
            Number of columns to be generated in the grid.

        grid_limit: float
            Range of the values in the grid generated.

        neg_allowed: np.ndarray
            Ensures if we want to include negative values in the grid or not.
            If True, the range is doubled.

        force_L1_norm: bool
            Ensures L1 normalization if set to True.
        """
        self.grid_size = grid_size
        self.grid_limit = grid_limit
        self.neg_allowed = neg_allowed if neg_allowed is not None else np.zeros((self.grid_size,), dtype=bool)
        self.force_L1_norm = force_L1_norm

    def generate_grid(self, constraint):
        """
        Generate a grid of lambda vectors based on the provided constraint.

        Parameters
        ----------
        constraint: Constraint
            Constraint object containing the basis for grid generation.

        Returns
        -------
        grid: pd.DataFrame
            Generated grid of lambda vectors.
        """
        self.dim = len(constraint.basis["+"].columns)
        coefs = self._generate_coefs()
        grid = constraint.basis["+"].dot(coefs["+"]) + constraint.basis["-"].dot(coefs["-"])
        return grid

    def _generate_coefs(self):
        np_grid_values = self._build_grid()
        grid_values = pd.DataFrame(np_grid_values[: self.grid_size]).T
        pos_grid_values = grid_values.clip(lower=0)
        neg_grid_values = -grid_values.clip(upper=0)
        lambda_vector = {"+": pos_grid_values, "-": neg_grid_values}
        return lambda_vector

    def _build_grid(self):
        max_value = self._get_true_dim()
        self.accumulator = []

        while len(self.accumulator) < self.grid_size:
            entry = np.zeros(self.dim)
            self._accumulate_integer_grid(0, max_value, entry)
            max_value += 1

        xs = np.array(self.accumulator) * self.grid_limit / max_value
        return xs

    def _accumulate_integer_grid(self, index, max_val, entry):
        if index == self.dim:
            self.accumulator.append(entry.copy())
            return

        if (index == self.dim - 1) and self.force_L1_norm:
            values = [-max_val, max_val] if self.neg_allowed[index] and max_val > 0 else [max_val]
        else:
            min_val = -max_val if self.neg_allowed[index] else 0
            values = range(min_val, max_val + 1)

        for current_value in values:
            entry[index] = current_value
            self._accumulate_integer_grid(index + 1, max_val - abs(current_value), entry)

    def _get_true_dim(self):
        true_dim = self.dim - 1 if self.force_L1_norm else self.dim
        n_units = (self.grid_size / (2 ** self.neg_allowed.sum())) ** (1 / true_dim) - 1
        return max(0, int(np.floor(n_units)))
