import numpy as np
import pandas as pd
from holisticai.bias.mitigation.inprocessing.commons._conventions import _ALL, _LABEL
from holisticai.bias.mitigation.inprocessing.commons._moments_utils import BaseMoment


class ErrorRate(BaseMoment):
    """
    Extend BaseMoment for error rate objective.
    A classifier :math:`h(X)` has the misclassification error equal to
    .. math::
      P[h(X) \ne Y]
    """

    def load_data(self, X, y, sensitive_features):
        super().load_data(X, y, sensitive_features)
        self.index = [_ALL]

    def signed_weights(self):
        """Return the signed weights."""
        return 2 * self.tags[_LABEL] - 1

    def gamma(self, predictor):
        """Return the gamma values for the given predictor."""
        pred = predictor(self.X)

        if isinstance(pred, np.ndarray):
            pred = np.squeeze(pred)

        error = pd.Series(data=(self.tags[_LABEL] - pred).abs().mean(), index=self.index)
        return error
