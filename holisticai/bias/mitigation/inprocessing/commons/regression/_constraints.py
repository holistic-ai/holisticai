import numpy as np
import pandas as pd

from .._conventions import _ALL, _EVENT, _GROUP_ID, _LABEL, _LOSS, _PREDICTION, _SIGNED
from .._moments_utils import BaseMoment, format_data


class RegressionConstraint(BaseMoment):
    """Constrain the mean loss or the worst-case loss by a group"""

    PROBLEM_TYPE = "regression"

    def __init__(self, loss, upper_bound=None, no_groups=False):
        """
        Parameters
        ----------
        loss : {SquareLoss, AbsoluteLoss}
            A loss object with an `eval` method, e.g. `SquareLoss` or
            `AbsoluteLoss`.
        upper_bound : float
            An upper bound on the loss, also referred to as :math:`\\zeta`;
            `upper_bound` is an optional argument that is not always
            required; default None
        no_groups : bool
            indicates whether to calculate the mean loss or group-level losses,
            default False, i.e., group-level losses are the default behavior
        """
        super().__init__()
        self.reduction_loss = loss
        self.upper_bound = upper_bound
        self.no_groups = no_groups

    def default_objective(self):
        """Return a default objective."""
        return MeanLoss(self.reduction_loss)

    def load_data(self, X, y, sensitive_features):
        params = format_data(y=y)
        y = params["y"]

        if self.no_groups:
            sensitive_features = np.zeros_like(y)

        # The following uses X and not X_train so that the estimators get X untouched
        super().load_data(X, y, sensitive_features=sensitive_features)
        event = pd.Series(data=_ALL, index=self.tags[_LABEL].index)
        self.tags[_EVENT] = event
        self.event_ids = np.sort(self.tags[_EVENT].dropna().unique())
        self.event_prob = self.tags[_EVENT].dropna().value_counts() / len(self.tags)

        self.group_prob = (
            self.tags.groupby(_GROUP_ID).size() / self.total_samples
        )  # self.tags[_GROUP_ID].dropna().value_counts() / len(self.tags)
        self.group_values = np.sort(self.tags[_GROUP_ID].unique())

        self.index = self.group_prob.index
        self.default_objective_lambda_vec = self.group_prob

        self._get_basis()

    def _get_basis(self):
        pos_basis = pd.DataFrame()
        neg_basis = pd.DataFrame()
        neg_basis_present = pd.Series(dtype="float64")
        zero_vec = pd.Series(0.0, self.index)
        i = 0
        for group in self.group_values:
            pos_basis[i] = zero_vec
            neg_basis[i] = zero_vec
            pos_basis[i][group] = 1
            neg_basis_present.at[i] = False
            i += 1

        self.neg_basis_present = neg_basis_present
        self.basis = {"+": pos_basis, "-": neg_basis}
        self.default_objective_lambda_vec = self.group_prob

    def gamma(self, predictor):
        """Calculate the degree to which constraints are currently violated by the predictor."""
        self.tags[_PREDICTION] = predictor(self.X)
        self.tags[_LOSS] = self.reduction_loss.eval(
            self.tags[_LABEL], self.tags[_PREDICTION]
        )
        expect_attr = self.tags.groupby(_GROUP_ID).mean()
        return expect_attr[_LOSS]

    def bound(self):
        """Return the vector of bounds.
        Returns
        -------
        pandas.Series
            A vector of bounds on group-level losses
        """
        if self.upper_bound is None:
            raise ValueError("No Upper Bound")
        return pd.Series(self.upper_bound, index=self.index)

    def project_lambda(self, lambda_vec):
        """Return the lambda values."""
        return lambda_vec

    def signed_weights(self, lambda_vec=None):
        """Return the signed weights."""
        if lambda_vec is None:
            adjust = pd.Series(1.0, index=self.index)
        else:
            adjust = lambda_vec / self.group_prob
        return self.tags.apply(lambda row: adjust[row[_GROUP_ID]], axis=1)


class MeanLoss(RegressionConstraint):
    """
    Moment for evaluating the mean loss.
    """

    def __init__(self, loss):
        super().__init__(loss, upper_bound=None, no_groups=True)


class BoundedGroupLoss(RegressionConstraint):
    """
    Moment for constraining the worst-case loss by a group.
    """

    def __init__(self, loss, *, upper_bound=None):
        super().__init__(loss, upper_bound=upper_bound, no_groups=False)
