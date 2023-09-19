import numpy as np
import pandas as pd

from ._conventions import _EVENT, _GROUP_ID, _LABEL, _SIGNED


def merge_columns(feature_columns):
    return pd.DataFrame(feature_columns).apply(
        lambda row: ",".join([str(r) for r in row.values]), axis=1
    )


def format_data(y=None):
    new_y = pd.Series(np.array(y).reshape(-1))
    return {"y": new_y}


class BaseMoment:
    """
    Base Moment

    Abstract class used to used to describe:
    - Optimization Objective
    - Fairness Constraints
    """

    params = ["X", "y", "sensitive_features", "control_features"]

    def save_params(self, *args):
        for name, value in zip(self.params, args):
            setattr(self, name, value)

    def load_data(self, X, y, sensitive_features):
        self.tags = pd.DataFrame()
        self.tags[_LABEL] = y
        self.tags[_GROUP_ID] = merge_columns(sensitive_features)
        self.save_params(X, y, sensitive_features)

    @property
    def total_samples(self):
        """Return the number of samples in the data."""
        return self.X.shape[0]

    @property
    def _y_as_series(self):
        """Return the y array as a :class:`~pandas.Series`."""
        return self.y

    def project_lambda(self, lambda_vec):
        """
        Projected lambda values.

        Returns
        -------
        Returns lambda which is guaranteed to lead to the same or higher value of the
        Lagrangian compared with lambda_vec for all possible choices of the classifier, h.
        """
        if self.ratio == 1.0:
            lambda_pos = lambda_vec["+"] - lambda_vec["-"]
            lambda_neg = -lambda_pos
            lambda_pos[lambda_pos < 0.0] = 0.0
            lambda_neg[lambda_neg < 0.0] = 0.0
            lambda_projected = pd.concat(
                [lambda_pos, lambda_neg],
                keys=["+", "-"],
                names=[_SIGNED, _EVENT, _GROUP_ID],
            )
            return lambda_projected
        return lambda_vec

    def bound(self):
        """
        Return bound vector.

        Returns
        -------
        pandas.Series
            a vector of bound values corresponding to all constraints

        """
        return pd.Series(self.eps, index=self.index)
