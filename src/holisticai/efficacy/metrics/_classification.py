import numpy as np
import pandas as pd
from holisticai.utils._validation import _multiclass_checks
from sklearn.metrics import mean_squared_error


def confusion_matrix(y_pred, y_true, classes=None, normalize=None):
    """
    Confusion Matrix.

    Description
    ----------
    This function computes the confusion matrix. The i,jth
    entry is the number of elements with predicted class i
    and true class j.

    Parameters
    ----------
    y_pred : array-like
        Prediction vector (categorical)
    y_true : array-like
        Target vector (categorical)
    classes (optional) : list
        The unique output classes in order
    normalize (optional) : None, 'pred' or 'class'
        According to which of pred or class we normalize

    Returns
    -------
    numpy ndarray
        Confusion Matrix : shape (num_classes, num_classes)

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import confusion_matrix
    >>> y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    >>> y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1])
    >>> confusion_matrix(y_pred, y_true, classes=[2, 1, 0])
        2    1    0
    2  1.0  1.0  1.0
    1  0.0  3.0  0.0
    0  1.0  1.0  2.0
    """
    # check and coerce inputs
    _, y_pred, y_true, _, classes = _multiclass_checks(
        p_attr=None,
        y_pred=y_pred,
        y_true=y_true,
        groups=None,
        classes=classes,
    )

    # variables
    num_classes = len(classes)
    class_dict = dict(zip(classes, range(num_classes)))

    # initialize the confusion matrix
    confmat = np.zeros((num_classes, num_classes))

    # loop over instances
    for x, y in zip(y_pred, y_true):
        # increment correct entry
        confmat[class_dict[x], class_dict[y]] += 1

    if normalize is None:
        pass

    elif normalize == "pred":
        confmat = confmat / np.sum(confmat, axis=1).reshape(-1, 1)

    elif normalize == "true":
        confmat = confmat / np.sum(confmat, axis=0).reshape(1, -1)

    else:
        raise ValueError('normalize should be one of None, "pred" or "true"')

    return pd.DataFrame(confmat, columns=classes).set_index(np.array(classes))


def classification_efficacy_metrics(y_pred, y_true=None, y_proba=None):
    """
    Classification efficacy metrics batch computation.

    Description
    -----------
    This function computes all the relevant classification efficacy metrics,
    and displays them as a pandas dataframe. It also includes a reference value for comparison.

    Parameters
    ----------
    y_pred : array-like
        Predictions vector (binary)
    y_true (optional) : numpy array
        Target vector (binary)
    y_proba (optional) : array-like
        Probability Matrix estimates (num_examples, 2)

    Returns
    -------
    pandas DataFrame
        Metrics | Values | Reference
    """
    from sklearn import metrics

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
        "Precision": {},
        "Recall": {},
        "F1-Score": {},
    }

    soft_perform = {"AUC": metrics.roc_auc_score, "Log Loss": metrics.log_loss}

    soft_hyper = {"AUC": {"average": "micro"}, "Log Loss": {}}

    ref_vals = {
        "Accuracy": 1,
        "Balanced Accuracy": 1,
        "Precision": 1,
        "Recall": 1,
        "F1-Score": 1,
        "AUC": 1,
        "Log Loss": 0,
    }
    metrics = [[pf, fn(y_pred, y_true, **hyper[pf]), ref_vals[pf]] for pf, fn in perform.items()]
    if y_proba is not None:
        opp_metrics = [[pf, fn(y_pred, y_true, **soft_hyper[pf]), ref_vals[pf]] for pf, fn in soft_perform.items()]
        metrics += opp_metrics

    return pd.DataFrame(metrics, columns=["Metric", "Value", "Reference"]).set_index("Metric")


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


def rmse_score(y_true, y_pred, **kargs):
    return np.sqrt(mean_squared_error(y_true, y_pred, **kargs))


def smape(y_true, y_pred):
    return 1.0 / len(y_true) * np.sum(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def regression_efficacy_metrics(y_pred, y_true):
    """
    Regression efficacy metrics batch computation.

    Description
    -----------
    This function computes all the relevant regression efficacy metrics,
    and displays them as a pandas dataframe. It also includes a reference value for comparison.

    Parameters
    ----------
    y_pred : array-like
        Predictions vector
    y_true (optional) : array-like
        Target vector

    Returns
    -------
    pandas DataFrame
        Metrics | Values | Reference
    """
    from sklearn import metrics

    perform = {
        # "Pseudo-R2": pr2_score,
        "RMSE": rmse_score,
        "MAE": metrics.mean_absolute_error,
        "MAPE": metrics.mean_absolute_percentage_error,
        "Max Error": metrics.max_error,
        "SMAPE": smape,
    }

    hyper = {
        # "Pseudo-R2": {},
        "RMSE": {"sample_weight": None, "multioutput": "uniform_average"},
        "MAE": {"sample_weight": None, "multioutput": "uniform_average"},
        "MAPE": {"sample_weight": None, "multioutput": "uniform_average"},
        "Max Error": {},
        "SMAPE": {},
    }

    ref_vals = {
        "RMSE": 0,
        "MAE": 0,
        "MAPE": 0,
        "Max Error": 0,
        "SMAPE": 0,
    }

    metrics = [[pf, fn(y_pred, y_true, **hyper[pf]), ref_vals[pf]] for pf, fn in perform.items()]

    return pd.DataFrame(metrics, columns=["Metric", "Value", "Reference"]).set_index("Metric")
