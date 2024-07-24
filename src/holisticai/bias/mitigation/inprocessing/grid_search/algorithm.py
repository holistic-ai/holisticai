import copy
from typing import Any

import pandas as pd
from holisticai.bias.mitigation.inprocessing.grid_search._grid_generator import GridGenerator
from sklearn.utils.parallel import Parallel, delayed


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
        estimator: Any,
        constraint: Any,
        constraint_weight: float = 0.5,
        grid_size: int = 20,
        grid_limit: int = 2,
        n_jobs=-1,
        verbose=0,
    ):
        """
        Init GridSearchAlgorithm object

        Parameters
        ----------
        estimator : object
            An estimator implementing methods :code:`fit(X, y, sample_weight=sample_weight)` and
            :code:`predict(X)`, where `X` is the matrix of features, `y` is the
            vector of labels, and `sample_weight` is a vector of weights.
            In binary classification labels `y` and predictions returned by
            :code:`predict(X)` are either 0 or 1.

        constraint : ClassificationConstraint
            The disparity constraint.

        constraint_weight : float
            Specifies the relative weight put on the constraint violation when selecting the
            best model. The weight placed on the error rate will be :code:`1-constraint_weight`.

        grid_size: int
            Number of columns to be generated in the grid.

        grid_limit : float
            Range of the values in the grid generated.

        verbose : int
            If >0, will show progress percentage.
        """
        self.estimator: Any = estimator
        self.constraint: Any = constraint
        self.objective: Any = constraint.default_objective()
        self.grid_limit: int = grid_limit
        self.grid_size: int = grid_size
        self.constraint_weight = constraint_weight
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: Any, y: Any, sensitive_features: Any):
        """
        Fit model using Grid Search Algorithm.

        Parameters
        ----------
        X : matrix-like
            Input matrix.

        y : numpy array
            Target vector.

        sensitive_features : numpy array
            Matrix where each column is a sensitive feature e.g. [col_1=group_a, col_2=group_b].

        Returns
        -------
        GridSearchAlgorithm
            The same object.
        """
        self._load_data(X, y, sensitive_features)
        grid = self._generate_grid()

        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self.evaluate_candidate)(
                X,
                grid[col_name],
            )
            for col_name in grid
        )

        results = pd.DataFrame(results)
        results["col_name"] = grid.columns
        best_idx = results["loss"].idxmin()
        best_colname = results["col_name"].iloc[best_idx]
        self.bet_predictor = self._fit_estimator(X, grid[best_colname])
        return self

    def evaluate_candidate(self, X, lambda_vec):
        current_estimator = self._fit_estimator(X, lambda_vec)

        def predict_fn(X: Any) -> Any:
            return current_estimator.predict(X)

        objective = self.objective.gamma(predict_fn).iloc[0]
        gamma = self.constraint.gamma(predict_fn)
        loss = (1 - self.constraint_weight) * objective + self.constraint_weight * gamma.max()
        return {"loss": loss, "lambda_vec": lambda_vec}

    def _load_data(self, X: Any, y: Any, sensitive_features: Any):
        self.constraint.load_data(X, y, sensitive_features)
        self.objective.load_data(X, y, sensitive_features)

    def _generate_grid(self) -> Any:
        neg_allowed = self.constraint.neg_basis_present
        objective_in_the_span = self.constraint.default_objective_lambda_vec is not None

        generator = GridGenerator(
            grid_size=self.grid_size,
            grid_limit=self.grid_limit,
            neg_allowed=neg_allowed,
            force_L1_norm=objective_in_the_span,
        )

        return generator.generate_grid(self.constraint)

    def _fit_estimator(self, X: Any, lambda_vec: Any):
        weights = self._compute_weights(lambda_vec)
        y_reduction, weights = self._get_reduction(weights)
        current_estimator = copy.deepcopy(self.estimator)
        current_estimator.fit(X, y_reduction, sample_weight=weights)
        return current_estimator

    def _compute_weights(self, lambda_vec: Any) -> Any:
        weights = self.constraint.signed_weights(lambda_vec)
        if not self.constraint.default_objective_lambda_vec:
            weights += self.objective.signed_weights()
        return weights

    def _get_reduction(self, weights: Any) -> tuple:
        if self.constraint.PROBLEM_TYPE == "classification":
            y_reduction = 1 * (weights > 0)
            weights = weights.abs()
        else:
            y_reduction = self.constraint.y_as_series
        return y_reduction, weights

    def predict(self, X: Any) -> Any:
        return self.bet_predictor.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self.bet_predictor.predict_proba(X)
