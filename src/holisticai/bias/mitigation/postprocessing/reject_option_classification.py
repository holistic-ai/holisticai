from __future__ import annotations

import itertools
import sys
from typing import Literal

import holisticai.bias.metrics as bias_metrics
import numpy as np
from holisticai.utils.transformers.bias import BMPostprocessing as BMPost
from sklearn.metrics import balanced_accuracy_score


def statistical_parity(group_a, group_b, y_pred, _):
    return bias_metrics.statistical_parity(group_a, group_b, y_pred)


class RejectOptionClassification(BMPost):
    """
    Reject option classification gives favorable outcomes (y=1) to unpriviliged groups and unfavorable outcomes (y=0) to\
    priviliged groups in a confidence band around the decision boundary with the highest uncertainty.

    Parameters
    ----------
    low_class_thresh : float
        Smallest classification threshold to use in the optimization.
        Should be between 0. and 1.
    low_class_thresh : float
        Smallest classification threshold to use in the optimization.
        Should be between 0. and 1.
    high_class_thresh : float
        Highest classification threshold to use in the optimization.
        Should be between 0. and 1.
    num_class_thresh : int
        Number of classification thresholds between low_class_thresh and high_class_thresh for the optimization
        search. Should be > 0.
    num_ROC_margin : int
        Number of relevant ROC margins to be used in the optimization search. Should be > 0.
    metric_name : str
        Name of the metric to use for the optimization. Allowed options are:
        "Statistical parity difference",
        "Average odds difference",
        "Equal opportunity difference".
    metric_ub : float
        Upper bound of constraint on the metric value
    metric_lb : float
        Lower bound of constraint on the metric value
    verbose : int
        If >0, will show progress percentage.

    References
    ----------
        .. [1] Kamiran, Faisal, Asim Karim, and Xiangliang Zhang. "Decision theory for discrimination-aware classification."\
        2012 IEEE 12th International Conference on Data Mining. IEEE, 2012.
    """

    ALLOWED_METRICS = Literal[
        "Statistical parity difference",
        "Average odds difference",
        "Equal opportunity difference",
    ]

    def __init__(
        self,
        low_class_thresh: float | None = 0.01,
        high_class_thresh: float | None = 0.99,
        num_class_thresh: int | None = 100,
        num_ROC_margin: int | None = 50,
        metric_name: str | None = "Statistical parity difference",
        metric_ub: float | None = 0.05,
        metric_lb: float | None = -0.05,
        num_workers: int | None = 4,
        verbose: int | None = 0,
    ):
        super().__init__()

        allowed_metrics = [
            "Statistical parity difference",
            "Average odds difference",
            "Equal opportunity difference",
        ]

        self.low_class_thresh = low_class_thresh
        self.high_class_thresh = high_class_thresh
        self.num_class_thresh = num_class_thresh
        self.num_ROC_margin = num_ROC_margin
        self.metric_name = metric_name
        self.metric_ub = metric_ub
        self.metric_lb = metric_lb
        self.num_workers = num_workers
        self.verbose = verbose

        self.classification_threshold = None
        self.ROC_margin = None

        if (
            (self.low_class_thresh < 0.0)
            or (self.low_class_thresh > 1.0)
            or (self.high_class_thresh < 0.0)
            or (self.high_class_thresh > 1.0)
            or (self.low_class_thresh >= self.high_class_thresh)
            or (self.num_class_thresh < 1)
            or (self.num_ROC_margin < 1)
        ):
            msg = "Input parameter values out of bounds"
            raise ValueError(msg)

        if metric_name not in allowed_metrics:
            msg = "metric name not in the list of allowed metrics"
            raise ValueError(msg)

    def fit(
        self,
        y: np.ndarray,
        y_proba: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Compute parameters for reject option classification strategy.

        Description
        ----------
        Evaluate the likelihood with different thresholds. Select the threshold and ROC margin \
        with the best fair metric value and highest balanced accuracy.

        Parameters
        ----------
        y : array-like
            Target vector (nb_examples,)
        y_proba : matrix-like
            Predicted probability matrix (num_examples, num_classes). The probability
            estimates must sum to 1 across the possible classes and each matrix value
            must be in the interval [0,1].
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        -------
            Self
        """

        params = self._load_data(y=y, y_proba=y_proba, group_a=group_a, group_b=group_b)

        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1
        y = params["y"]
        likelihoods = params["y_score"]
        y_proba = params["y_proba"]

        class_thresholds = np.linspace(self.low_class_thresh, self.high_class_thresh, self.num_class_thresh)

        if self.metric_name == "Statistical parity difference":
            fair_metric = statistical_parity

        elif self.metric_name == "Average odds difference":
            fair_metric = bias_metrics.average_odds_diff

        elif self.metric_name == "Equal opportunity difference":
            fair_metric = bias_metrics.equal_opportunity_diff

        else:
            msg = "metric name not in the list of allowed metrics"
            raise ValueError(msg)

        args_iterator = [
            (
                fair_metric,
                y,
                likelihoods,
                group_a,
                group_b,
                class_threshold,
                self.num_ROC_margin,
            )
            for class_threshold in class_thresholds
        ]

        # Serial
        # configurations = []
        # self.num_cases = len(args_iterator)
        # for i, argv in enumerate(args_iterator):
        #    configurations.append(_evaluate_threshold(*argv))
        #    self._log_progres(i)

        # Pool
        from multiprocessing import Pool

        with Pool(self.num_workers) as p:
            configurations = list(p.starmap(_evaluate_threshold, args_iterator))

        # Joblib
        # from joblib import Parallel, delayed
        # configurations = Parallel(n_jobs=-1)(
        #    delayed(_evaluate_threshold)(*args) for args in args_iterator
        # )

        configurations = list(itertools.chain.from_iterable(configurations))

        selected_configurations = list(
            filter(
                lambda c: (c["fair_score"] >= self.metric_lb) and (c["fair_score"] <= self.metric_ub),
                configurations,
            )
        )

        if any(selected_configurations):
            best_config = max(selected_configurations, key=lambda c: c["balanced_accurracy"])
        else:
            best_config = min(configurations, key=lambda c: abs(c["fair_score"]))

        self.ROC_margin = best_config["roc_margin"]
        self.classification_threshold = best_config["class_thresh"]

        return self

    def transform(
        self,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        group_a: np.ndarray,
        group_b: np.ndarray,
    ):
        """
        Obtain fair predictions using the ROC method.

        Description
        ----------
        Predict the output using the fitted threshold and ROC margin.

        Parameters
        ----------
        y_pred : array-like
            Predicted vector (nb_examples,)
        y_proba : matrix-like
            Predicted probability matrix (num_examples, num_classes). The probability\
            estimates must sum to 1 across the possible classes and each matrix value\
            must be in the interval [0,1].
        group_a : array-like
            Group membership vector (binary)
        group_b : array-like
            Group membership vector (binary)

        Returns
        -------
        dict
            A dictionary of new predictions
        """
        params = self._load_data(y_pred=y_pred, y_proba=y_proba, group_a=group_a, group_b=group_b)

        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1
        y_pred = params["y_pred"]
        y_score = params["y_score"]

        new_y_pred = predict(
            y_score > self.classification_threshold,
            y_score,
            group_a,
            group_b,
            self.classification_threshold,
            self.ROC_margin,
        )
        new_y_score = y_score.copy()

        return {
            "y_pred": new_y_pred,
            "y_score": new_y_score,
        }

    def _log_progres(self, i):
        if self.verbose > 0:
            sys.stdout.write(f"\rProgress: {i+1}/{self.num_cases}")
            sys.stdout.flush()


def _evaluate_threshold(fair_metric, labels, likelihoods, group_a, group_b, class_thresh, num_roc_margin):
    base_predictions = np.where(likelihoods > class_thresh, 1, 0)
    high_roc_margin = class_thresh if class_thresh <= 0.5 else 1.0 - class_thresh
    roc_margins = np.linspace(0.0, high_roc_margin, num_roc_margin)
    configurations = []
    for roc_margin in roc_margins:
        prediction = predict(base_predictions, likelihoods, group_a, group_b, class_thresh, roc_margin)
        configuration = {
            "class_thresh": class_thresh,
            "roc_margin": roc_margin,
            "balanced_accurracy": balanced_accuracy_score(labels, prediction),
            "fair_score": fair_metric(group_b, group_a, prediction, labels),
        }
        configurations.append(configuration)
    return configurations


def predict(predictions, likelihoods, group_a, group_b, threshold, roc_margin):
    """
    Predict the output using the ROC method.

    Description
    ----------
    Predict the output using the fitted threshold and ROC margin.

    Parameters
    ----------
    predictions : array-like
        Predicted vector (nb_examples,)
    likelihoods : matrix-like
        Predicted probability matrix (num_examples, num_classes). The probability\
        estimates must sum to 1 across the possible classes and each matrix value\
        must be in the interval [0,1].
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    threshold : float
        float value to discriminate between 0 and 1
    roc_margin : float
        float value to determine the margin around the decision boundary

    Returns
    -------
    array-like
        A new array of predictions
    """
    # Indices of critical region around the classification boundary
    upper_threshold = threshold + roc_margin
    lower_threshold = threshold - roc_margin
    crit_region_inds = np.logical_and(likelihoods <= upper_threshold, likelihoods > lower_threshold)

    # New, fairer labels
    new_predictions = predictions.copy()
    new_predictions[np.logical_and(crit_region_inds, group_a)] = 1
    new_predictions[np.logical_and(crit_region_inds, group_b)] = 0
    return new_predictions
