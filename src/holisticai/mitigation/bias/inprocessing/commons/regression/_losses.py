import numpy as np


class SquareLoss:
    """Class to evaluate the square loss.
    Read more in the :ref:`User Guide <constraints_regression>`.
    """

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.min = 0
        self.max = (max_val - min_val) ** 2

    def eval(self, y_true, y_pred):
        """Evaluate the square loss for the given set of true and predicted values."""
        y_true = np.clip(y_true, self.min_val, self.max_val)
        y_pred = np.clip(y_pred, self.min_val, self.max_val)
        return (y_true - y_pred) ** 2


class AbsoluteLoss:
    """Class to evaluate absolute loss.
    Read more in the :ref:`User Guide <constraints_regression>`.
    """

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.min = 0
        self.max = np.abs(max_val - min_val)

    def eval(self, y_true, y_pred):
        """Evaluate the absolute loss for the given set of true and predicted values."""
        y_true = np.clip(y_true, self.min_val, self.max_val)
        y_pred = np.clip(y_pred, self.min_val, self.max_val)
        return np.abs(y_true - y_pred)


class ZeroOneLoss(AbsoluteLoss):
    """Class to evaluate a zero-one loss.
    Read more in the :ref:`User Guide <constraints_regression>`.
    """

    def __init__(self):
        super().__init__(0, 1)
