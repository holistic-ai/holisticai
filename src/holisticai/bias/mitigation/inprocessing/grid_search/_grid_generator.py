from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GridGenerator:
    """
    Create a grid using a constraint
    """

    def __init__(
        self,
        grid_size: int = 5,
        grid_limit: float = 2.0,
        neg_allowed: Optional[np.ndarray] = None,
        force_L1_norm: bool = True,
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

    def generate_grid(self, constraint, grid_offset=None):
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
        if grid_offset is None:
            grid_offset = pd.Series(0, index=constraint.basis["+"].index)

        self.dim = len(constraint.basis["+"].columns)
        coefs = self._generate_coefs()
        grid = constraint.basis["+"].dot(coefs["+"]) + constraint.basis["-"].dot(coefs["-"])
        return grid.add(grid_offset, axis="index")

    def _generate_coefs(self):
        n_units = self._get_true_dim()
        np_grid_values = self._build_grid(n_units)
        pos_coefs = pd.DataFrame(np_grid_values[: self.grid_size]).T * (float(self.grid_limit) / float(n_units))
        neg_coefs = -pos_coefs.copy()
        pos_coefs[pos_coefs < 0] = 0.0
        neg_coefs[neg_coefs < 0] = 0.0
        lambda_vector = {"+": pos_coefs, "-": neg_coefs}
        return lambda_vector

    def _build_grid(self, n_units):
        """
        Build a grid by accumulating points until grid_size is reached.
        """

        self.accumulator = []
        while len(self.accumulator) < self.grid_size:
            self.entry = np.zeros(self.dim)
            self._accumulate_integer_grid(0, n_units)
            n_units += 1

        # If the desired grid size is not reached after max_iterations, raise a warning.
        if len(self.accumulator) < self.grid_size:
            logger.warning(
                f"Warning: The desired grid size was not reached. {len(self.accumulator)} points were generated."
            )

        xs = np.array(self.accumulator) * self.grid_limit / n_units
        return xs

    def _accumulate_integer_grid(self, index, max_val):
        """
        Recursive function to generate grid values.
        """
        if index == self.dim:
            self.accumulator.append(self.entry.copy())
            return

        if (index == self.dim - 1) and self.force_L1_norm:
            values = [-max_val, max_val] if self.neg_allowed[index] and max_val > 0 else [max_val]
        else:
            min_val = -max_val if self.neg_allowed[index] else 0
            values = range(min_val, max_val + 1)

        for current_value in values:
            self.entry[index] = current_value
            self._accumulate_integer_grid(index + 1, max_val - abs(current_value))

    def _get_true_dim(self):
        """
        Get the true dimension of the grid, adjusting if force_L1_norm is True.
        """
        true_dim = self.dim - 1 if self.force_L1_norm else self.dim
        # Adjust the calculation of n_units to avoid dimensionality problems.
        n_units = (float(self.grid_size) / (2 ** self.neg_allowed.sum())) ** (1 / true_dim) - 1
        return max(1, int(np.floor(n_units)))
