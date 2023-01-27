import copy
import sys
from typing import Optional

import numpy as np

from ._grid_generator import GridGenerator


class GridSearchAlgorithm:
    """
    Estimator to perform a grid search given a blackbox estimator algorithm.
    The approach used is taken from section 3.4 of "A reductions approach to fair classification"
    Enabled for more than two sensitive feature values

    Reference
    ---------

    Agarwal, Alekh, et al. "A reductions approach to fair classification."
    International Conference on Machine Learning. PMLR, 2018.
    """

    def __init__(
        self,
        estimator,
        constraint,
        constraint_weight: Optional[float] = 0.5,
        grid_size: Optional[int] = 20,
        grid_limit: Optional[int] = 2,
        verbose: Optional[int] = 0,
    ):
        """
        Init GridSearchAlgorithm object

        Parameters
        ----------

        estimator :
            An estimator implementing methods :code:`fit(X, y, sample_weight=sample_weight)` and
            :code:`predict(X)`, where `X` is the matrix of features, `y` is the
            vector of labels, and `sample_weight` is a vector of weights.
            In binary classification labels `y` and predictions returned by
            :code:`predict(X)` are either 0 or 1.

        generator : float
            grid generator utility

        constraint : ClassificationContraint
            The disparity constraint

        constraint_weight : float
            Specifies the relative weight put on the constraint violation when selecting the
            best model. The weight placed on the error rate will be :code:`1-constraint_weight`

        grid_size: int
            number of columns to be generated in the grid.

        grid_limit : float
            range of the values in the grid generated.

        verbose : int
            If >0, will show progress percentage.
        """

        self.constraint = constraint
        self.estimator = estimator
        self.objective = constraint.default_objective()
        self.grid_limit = grid_limit
        self.grid_size = grid_size
        self.monitor = Monitor(constraint_weight=constraint_weight, verbose=verbose)

    def fit(self, X, y, sensitive_features):
        """
        Fit model using Grid Search Algorithm.

        Parameters
        ----------

        X : matrix-like
            input matrix

        y_true : numpy array
            target vector

        sensitive_features : numpy array
            Matrix where each columns is a sensitive feature e.g. [col_1=group_a, col_2=group_b]

        Returns
        -------
        the same object
        """
        self.constraint.load_data(X, y, sensitive_features)
        self.objective.load_data(X, y, sensitive_features)

        neg_allowed = self.constraint.neg_basis_present
        objective_in_the_span = self.constraint.default_objective_lambda_vec is not None

        self.generator = GridGenerator(
            grid_size=self.grid_size,
            grid_limit=self.grid_limit,
            neg_allowed=neg_allowed,
            force_L1_norm=objective_in_the_span,
        )

        grid = self.generator.generate_grid(self.constraint)
        self.monitor.total_steps = grid.shape[1]
        for (_, lambda_vec) in grid.iteritems():

            weights = self.constraint.signed_weights(lambda_vec)
            if not objective_in_the_span:
                weights += self.objective.signed_weights()

            if self.constraint.PROBLEM_TYPE == "classification":
                y_reduction = 1 * (weights > 0)
                weights = weights.abs()
            else:
                y_reduction = self.constraint._y_as_series

            current_estimator = copy.deepcopy(self.estimator)

            current_estimator.fit(X, y_reduction, sample_weight=weights)

            predict_fn = lambda X: current_estimator.predict(X)
            objective_ = self.objective.gamma(predict_fn)[0]
            gamma_ = self.constraint.gamma(predict_fn)

            self.monitor.save(lambda_vec, current_estimator, objective_, gamma_)
            self.monitor.log_progress()

        self.best_idx_ = self.monitor.get_best_idx()

        return self

    def predict(self, X):
        return self.monitor.predictors_[self.best_idx_].predict(X)

    def predict_proba(self, X):
        return self.monitor.predictors_[self.best_idx_].predict_proba(X)


class Monitor:
    """Monitor class used to store historical data"""

    def __init__(self, constraint_weight, verbose):
        self.predictors_ = []
        self.lambda_vecs_ = []
        self.objectives_ = []
        self.gammas_ = []
        self.losses = []
        self.constraint_weight = float(constraint_weight)
        self.objective_weight = 1.0 - constraint_weight
        self.verbose = verbose
        self.step = 0

    def loss_fct(self, objective, gamma):
        return self.objective_weight * objective + self.constraint_weight * gamma.max()

    def save(self, lambda_vec, current_estimator, objective, gamma):

        self.predictors_.append(current_estimator)
        self.losses.append(self.loss_fct(objective, gamma))
        self.objectives_.append(objective)
        self.gammas_.append(gamma)
        self.lambda_vecs_.append(lambda_vec)

    def get_best_idx(self):
        return self.losses.index(min(self.losses))

    def log_progress(self):
        self.step += 1
        if self.verbose:
            sys.stdout.write(
                f"\r{self.step}/{self.total_steps}\tloss (best):{min(self.losses):.4f}"
            )
            sys.stdout.flush()
