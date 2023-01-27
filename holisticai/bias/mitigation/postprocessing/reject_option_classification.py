import itertools
import sys
from typing import Optional, Tuple, Union

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from holisticai.bias import metrics
from holisticai.utils.transformers.bias import BMPostprocessing as BMPost


class RejectOptionClassification(BMPost):
    """
    Reject option classification gives favorable outcomes (y=1) to unpriviliged groups and unfavorable outcomes (y=0)to
    priviliged groups in a confidence band around the decision boundary with the highest uncertainty.

    References:
    Kamiran, Faisal, Asim Karim, and Xiangliang Zhang. "Decision theory for discrimination-aware classification."
    2012 IEEE 12th International Conference on Data Mining. IEEE, 2012.
    """

    ALLOWED_METRICS = [
        "Statistical parity difference",
        "Average odds difference",
        "Equal opportunity difference",
    ]

    def __init__(
        self,
        low_class_thresh: Optional[float] = 0.01,
        high_class_thresh: Optional[float] = 0.99,
        num_class_thresh: Optional[int] = 100,
        num_ROC_margin: Optional[int] = 50,
        metric_name: Optional[str] = "Statistical parity difference",
        metric_ub: Optional[float] = 0.05,
        metric_lb: Optional[float] = -0.05,
        num_workers: Optional[int] = 8,
        verbose: Optional[int] = 0,
    ):
        """
        Create a Reject Option Classification Post-processing instance.

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

        """
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

            raise ValueError("Input parameter values out of bounds")

        if metric_name not in allowed_metrics:
            raise ValueError("metric name not in the list of allowed metrics")

    def fit(
        self,
        y_true: np.ndarray,
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
        y_true : array-like
            Target vector (nb_examlpes,)        
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

        params = self._load_data(
            y_true=y_true, y_proba=y_proba, group_a=group_a, group_b=group_b
        )

        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1
        y_true = params["y_true"]
        likelihoods = params["y_score"]
        y_proba = params["y_proba"]

        class_thresholds = np.linspace(
            self.low_class_thresh, self.high_class_thresh, self.num_class_thresh
        )

        args_iterator = [
            (
                self.metric_name,
                y_true,
                likelihoods,
                group_a,
                group_b,
                class_threshold,
                self.num_ROC_margin,
            )
            for class_threshold in class_thresholds
        ]

        # Serial
        configurations = []
        self.num_cases = len(args_iterator)
        for i, argv in enumerate(args_iterator):
            configurations.append(evaluate_threshold(*argv))
            self._log_progres(i)

        # Pool
        # from multiprocessing import Pool
        # with Pool(self.num_workers) as p:
        #    configurations = list(p.starmap(evaluate_threshold, args_iterator))

        # Joblib
        # from joblib import Parallel, delayed
        # configurations = Parallel(n_jobs=self.num_workers, verbose=0)(
        #    delayed(evaluate_threshold)(*args) for args in args_iterator
        # )

        configurations = list(itertools.chain.from_iterable(configurations))

        selected_configurations = list(
            filter(
                lambda c: (c["fair_metric"] >= self.metric_lb)
                and (c["fair_metric"] <= self.metric_ub),
                configurations,
            )
        )

        if any(selected_configurations):
            best_config = max(
                selected_configurations, key=lambda c: c["balanced_accurracy"]
            )
        else:
            best_config = min(configurations, key=lambda c: abs(c["fair_metric"]))

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
            Predicted vector (nb_examlpes,)
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
        dictionnary with new predictions
        """
        params = self._load_data(
            y_pred=y_pred, y_proba=y_proba, group_a=group_a, group_b=group_b
        )

        group_a = params["group_a"] == 1
        group_b = params["group_b"] == 1
        y_pred = params["y_pred"]
        y_score = params["y_score"]

        new_y_pred = predict(
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


def evaluate_threshold(
    metric_name, labels, likelihoods, group_a, group_b, class_thresh, num_roc_margin
):
    configurations = []
    high_roc_margin = class_thresh if class_thresh <= 0.5 else 1.0 - class_thresh
    for roc_margin in np.linspace(0.0, high_roc_margin, num_roc_margin):
        # Predict using the current threshold and margin
        predictions = predict(likelihoods, group_a, group_b, class_thresh, roc_margin)

        if metric_name == "Statistical parity difference":
            fair_metric = metrics.statistical_parity(group_b, group_a, predictions)

        elif metric_name == "Average odds difference":
            fair_metric = metrics.average_odds_diff(
                group_b, group_a, predictions, labels
            )

        elif metric_name == "Equal opportunity difference":
            fair_metric = metrics.equal_opportunity_diff(
                group_b, group_a, predictions, labels
            )

        configurations.append(
            {
                "class_thresh": class_thresh,
                "roc_margin": roc_margin,
                "balanced_accurracy": balanced_accuracy_score(labels, predictions),
                "fair_metric": fair_metric,
            }
        )
    return configurations


def predict(likelihoods, group_a, group_b, threshold, roc_margin):
    # Indices of critical region around the classification boundary
    upper_threshold = threshold + roc_margin
    lower_threshold = threshold - roc_margin
    crit_region_inds = np.logical_and(
        likelihoods <= upper_threshold, likelihoods > lower_threshold
    )

    # New, fairer labels
    new_predictions = np.where(likelihoods > threshold, 1, 0)
    new_predictions[np.logical_and(crit_region_inds, group_a)] = 1
    new_predictions[np.logical_and(crit_region_inds, group_b)] = 0
    return new_predictions
