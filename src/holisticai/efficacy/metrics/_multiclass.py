import numpy as np
import pandas as pd


def is_numeric(obj):
    attrs = ["__add__", "__sub__", "__mul__", "__truediv__", "__pow__"]
    return all(hasattr(obj, attr) for attr in attrs)


def multiclassification_efficacy_metrics(y_pred, y_true=None, y_proba=None, classes=None, by_class=False):
    """
    Multiclassification efficacy metrics batch computation.

    Description
    -----------
    This function computes all the relevant multiclassification efficacy metrics,
    and displays them as a pandas dataframe. It also includes a reference value for comparison.

    Parameters
    ----------
    y_pred : array-like
        Predictions vector
    y_true (optional) : array-like
        Target vector
    y_proba (optional) : array-like
        Probability Matrix estimates (num_examples, num_classes)

    Returns
    -------
    pandas DataFrame
        Metrics | Values | Reference
    """
    from sklearn import metrics

    if classes is None:
        classes = sorted(np.unique(y_pred))
        if y_proba is not None:
            assert all(
                is_numeric(c) for c in classes
            ), "to evaluate y_proba outputs, parameter `classes` must be passed."

    def convert_to_numeric_array(y, classes):
        if len(set(classes).intersection(set(np.unique(y)))) == len(classes):
            y = pd.Series(list(y)).replace(dict(zip(classes, range(len(classes)))))
        return y

    y_pred = convert_to_numeric_array(y_pred, classes)
    y_true = convert_to_numeric_array(y_true, classes)

    n_classes = len(classes)

    perform = {
        "Accuracy": metrics.accuracy_score,
        "Balanced Accuracy": metrics.balanced_accuracy_score,
        "Precision": metrics.precision_score,
        "Recall": metrics.recall_score,
        "F1-Score": metrics.f1_score,
    }

    hyper = {
        "Accuracy": {},
        "Balanced Accuracy": {},
        "Precision": {"average": "micro"},
        "Recall": {"average": "micro"},
        "F1-Score": {"average": "micro"},
    }

    soft_perform = {"AUC": metrics.roc_auc_score, "Log Loss": metrics.log_loss}

    soft_hyper = {
        "AUC": {"average": "macro", "multi_class": "ovr"},
        "Log Loss": {},
    }

    ref_vals = {
        "Accuracy": 1,
        "Balanced Accuracy": 1,
        "Precision": 1,
        "Recall": 1,
        "F1-Score": 1,
        "AUC": 1,
        "Log Loss": 0,
    }

    detailed_hyper = {
        "Precision": {"average": None},
        "Recall": {"average": None},
        "F1-Score": {"average": None},
        "AUC": {"average": None, "multi_class": "ovr"},
    }

    def process_value(metrics_dict, hyper_dict, y_pred, y_true):
        values_list = []
        for pf, fn in metrics_dict.items():
            result = fn(y_true, y_pred, **hyper_dict[pf])
            values_list.append(result)
        return values_list

    def process_metrics(values_list, metrics_dict, y_pred, y_true_oh, n_classes, detailed_hyper):
        metrics_list = []
        for value, (pf, fn) in zip(values_list, metrics_dict.items()):
            result = [value]
            if pf in detailed_hyper:
                result += list(fn(y_true_oh, y_pred, **detailed_hyper[pf]))
            else:
                result += [""] * n_classes
            metrics_list.append(result)
        return metrics_list

    metrics_1 = process_value(perform, hyper, y_pred, y_true)
    if by_class:
        metrics_1 = process_metrics(metrics_1, perform, y_pred, y_true, n_classes, detailed_hyper)
        metrics = [[pf, *metric, ref_vals[pf]] for metric, (pf, fn) in zip(metrics_1, perform.items())]
    else:
        metrics = [[pf, metric, ref_vals[pf]] for metric, (pf, fn) in zip(metrics_1, perform.items())]

    if y_proba is not None:
        y_true_oh = y_true.replace(dict(zip(classes, range(len(classes)))))
        metrics_2 = process_value(soft_perform, soft_hyper, y_proba, y_true_oh)
        if by_class:
            metrics_2 = process_metrics(metrics_2, soft_perform, y_proba, y_true_oh, n_classes, detailed_hyper)
            metrics_2 = [[pf, *metric, ref_vals[pf]] for metric, (pf, fn) in zip(metrics_2, soft_perform.items())]
        else:
            metrics_2 = [[pf, metric, ref_vals[pf]] for metric, (pf, fn) in zip(metrics_2, soft_perform.items())]
        metrics += metrics_2

    values_cols = ["Value"]
    if by_class:
        values_cols += classes

    return pd.DataFrame(metrics, columns=["Metric", *values_cols, "Reference"]).set_index("Metric")
