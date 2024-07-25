from __future__ import annotations

import numbers

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.parallel import Parallel, delayed

from holisticai.utils._definitions import ModelProxy, PartialDependence


def get_partial_dependence(
    estimator,
    x,
    features,
    sample_weight=None,
    categorical_features=None,
    feature_names=None,
    response_method="auto",
    grid_resolution=100,
    percentiles=(0.05, 0.95),
    method="auto",
    n_jobs=None,
    verbose=0,
    kind="average",
    subsample=1000,
) -> list[dict[str, np.ndarray]]:
    # expand kind to always be a list of str
    kind_ = [kind] * len(features) if isinstance(kind, str) else kind
    if len(kind_) != len(features):
        msg = (
            "When `kind` is provided as a list of strings, it should contain "
            f"as many elements as `features`. `kind` contains {len(kind_)} "
            f"element(s) and `features` contains {len(features)} element(s)."
        )
        raise ValueError(msg)

    if categorical_features is None:
        is_categorical = [(False,) for fxs in features]
    else:
        # we need to create a boolean indicator of which features are
        # categorical from the categorical_features list.
        categorical_features = np.asarray(categorical_features)
        is_categorical = [(categorical_features[fx],) for fx in features]

        for is_cat, kind_plot in zip(is_categorical, kind_):
            if any(is_cat) and kind_plot != "average":
                msg = "It is not possible to display individual effects for" " categorical features."
                raise ValueError(msg)

    if isinstance(subsample, numbers.Integral):
        if subsample <= 0:
            msg = f"When an integer, subsample={subsample} should be positive."
            raise ValueError(msg)

    elif isinstance(subsample, numbers.Real) and (subsample <= 0 or subsample >= 1):
        msg = f"When a floating-point, subsample={subsample} should be in " "the (0, 1) range."
        raise ValueError(msg)

    # compute predictions and/or averaged predictions
    return Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(partial_dependence)(
            estimator,
            x,
            fxs,
            sample_weight=sample_weight,
            feature_names=feature_names,
            categorical_features=categorical_features,
            response_method=response_method,
            method=method,
            grid_resolution=grid_resolution,
            percentiles=percentiles,
            kind=kind_plot,
        )  # type: ignore
        for kind_plot, fxs in zip(kind_, features)
    )


class SKLearnModel:
    # Skip pydantic Basemodel (necessary for validation)

    def __init__(self, predict, predict_proba, fit, score, classes, estimator_type):
        self.predict = predict
        self.fit = fit
        self.score = score
        self.foo_ = "foo_"  # necessary for partial_dependency_function validation
        self._estimator_type = estimator_type
        if predict_proba is not None:
            self.predict_proba = predict_proba
        if classes is not None:
            self.classes_ = classes

    def __sklearn_is_fitted__(self):
        return True


def wrap_sklearn_binary_model(classes: list, predict: callable, predict_proba: callable | None = None):
    def fit(x, y):  # noqa: ARG001
        pass

    def score(x, y):
        pred = predict(x)
        return accuracy_score(y_true=y, y_pred=pred)

    def predict_proba_fn(x):
        if predict_proba is None:
            return None
        prob = predict_proba(x)
        return np.stack([1 - prob, prob], axis=1)

    return SKLearnModel(
        predict=predict,
        predict_proba=predict_proba_fn,
        fit=fit,
        score=score,
        classes=classes,
        estimator_type="classifier",
    )


def wrap_sklearn_multi_classification_model(classes: list, predict: callable, predict_proba: callable | None = None):
    def fit(x, y):  # noqa: ARG001
        pass

    def score(x, y):
        pred = predict(x)
        return accuracy_score(y_true=y, y_pred=pred)

    def predict_proba_fn(x):
        return None if predict_proba is None else predict_proba(x)

    return SKLearnModel(
        predict=predict,
        predict_proba=predict_proba_fn,
        fit=fit,
        score=score,
        classes=classes,
        estimator_type="classifier",
    )


def wrap_sklearn_regression_model(predict: callable):
    def fit(x, y):  # noqa: ARG001
        pass

    def score(x, y):
        pred = predict(x)
        return r2_score(y_true=y, y_pred=pred)

    return SKLearnModel(
        predict=predict,
        predict_proba=None,
        fit=fit,
        score=score,
        classes=None,
        estimator_type="regressor",
    )


def wrap_sklearn_model(proxy: ModelProxy):
    learning_task = proxy.learning_task
    if learning_task == "binary_classification":
        return wrap_sklearn_binary_model(
            classes=proxy.classes,
            predict=proxy.predict,
            predict_proba=proxy.predict_proba,
        )
    if learning_task == "regression":
        return wrap_sklearn_regression_model(predict=proxy.predict)
    if learning_task == "multi_classification":
        return wrap_sklearn_multi_classification_model(
            classes=proxy.classes,
            predict=proxy.predict,
            predict_proba=proxy.predict_proba,
        )
    return None


def compute_partial_dependence(X: pd.DataFrame, features: list[str], proxy: ModelProxy) -> PartialDependence:
    supported_learning_tasks = ["binary_classification", "regression", "multi_classification"]
    if proxy.learning_task not in supported_learning_tasks:
        raise ValueError(f"Learning task {proxy.learning_task} is not supported for partial dependence computation")

    model = wrap_sklearn_model(proxy)
    feature_names = np.array(X.columns)
    method = "auto"
    response_method = "auto"
    grid_resolution = 50
    feature_index = [np.where(feature_names == f)[0][0] for f in features]
    percentiles = (0.05, 0.95) if proxy.learning_task == "regression" else (0, 1)
    partial_dependence = get_partial_dependence(
        model,  # type: ignore
        X,
        features=feature_index,
        kind="both",
        method=method,
        response_method=response_method,
        grid_resolution=grid_resolution,
        percentiles=percentiles,
        n_jobs=1,
    )

    if proxy.learning_task in ["binary_classification", "multi_classification"]:
        nb_classes = len(proxy.classes)
        new_partial_dependence = []
        for c in range(nb_classes):
            part_dep_feat = []
            for p in partial_dependence:
                assert nb_classes == len(p["individual"])
                newp = p.copy()
                newp["individual"] = p["individual"][c][np.newaxis]
                newp["average"] = p["average"][c][np.newaxis]
                part_dep_feat.append(newp)
            new_partial_dependence.append(part_dep_feat)
        return PartialDependence(values=new_partial_dependence)
    return PartialDependence(values=[partial_dependence])
