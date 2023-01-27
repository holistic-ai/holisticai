from typing import Optional

import numpy as np
import pandas as pd

from ..commons._conventions import *


class GridGenerator:
    """
    Create a grid using a constraint
    """

    def __init__(
        self,
        grid_size: Optional[int] = 5,
        grid_limit: Optional[int] = 2.0,
        neg_allowed: Optional[pd.DataFrame] = None,
        force_L1_norm: Optional[bool] = True,
    ):
        """
        Initialize Grid Generator

        Parameters
        ----------
        grid_size: int
            number of columns to be generated in the grid.

        grid_limit : float
            range of the values in the grid generated.

        neg_values: bool
            ensures if we want to include negative values in the grid or not.
            If True, the range is doubled.
        """
        self.grid_size = grid_size
        self.grid_limit = grid_limit
        self.neg_allowed = neg_allowed
        self.force_L1_norm = force_L1_norm

    def generate_grid(self, constraint):
        # Generate lambda vectors for each event
        self.dim = len(constraint.basis["+"].columns)
        coefs = self._generate_coefs()
        # Convert the grid of basis coefficients into a grid of lambda vectors
        grid = constraint.basis["+"].dot(coefs["+"]) + constraint.basis["-"].dot(
            coefs["-"]
        )
        return grid

    def _get_true_dim(self):
        if self.force_L1_norm:
            true_dim = self.dim - 1
        else:
            true_dim = self.dim

        n_units = (float(self.grid_size) / (2.0 ** self.neg_allowed.sum())) ** (
            1.0 / true_dim
        ) - 1
        n_units = int(np.floor(n_units))
        if n_units < 0:
            n_units = 0

        return n_units

    def _build_grid(self):
        """Create an integer grid"""
        max_value = self._get_true_dim()
        self.accumulator = []
        self.entry = np.zeros(self.dim)
        self._accumulate_integer_grid(0, max_value)
        xs = np.array(self.accumulator)
        xs = xs * self.grid_limit / max_value

        """
        min_value = 0
        if self.neg_values:
            max_value = (max_value - 1 + 2 - 1) // 2
            min_value = -max_value
        xs = [np.arange(min_value, max_value + 1) for _ in range(nb_events)]
        xs = np.meshgrid(*xs)
        xs = np.stack([x.reshape(-1) for x in xs], axis=1)
        xs = xs * self.grid_limit / max_value
        """
        return xs

    def _accumulate_integer_grid(self, index, max_val):
        if index == self.dim:
            self.accumulator.append(self.entry.copy())
        else:
            if (index == self.dim - 1) and (self.force_L1_norm):
                if self.neg_allowed[index] and max_val > 0:
                    values = [-max_val, max_val]
                else:
                    values = [max_val]
            else:
                min_val = -max_val if self.neg_allowed[index] else 0
                values = range(min_val, max_val + 1)

            for current_value in values:
                self.entry[index] = current_value
                self._accumulate_integer_grid(index + 1, max_val - abs(current_value))

    def _generate_coefs(self):
        np_grid_values = self._build_grid()
        grid_values = pd.DataFrame(np_grid_values[: self.grid_size]).T
        pos_grid_values = grid_values.copy()
        neg_grid_values = -grid_values.copy()
        pos_grid_values[grid_values < 0] = 0
        neg_grid_values[grid_values < 0] = 0
        lambda_vector = {"+": pos_grid_values, "-": neg_grid_values}
        return lambda_vector
