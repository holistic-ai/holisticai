import warnings

from sklearn.linear_model import LogisticRegression, LinearRegression

from holisticai.pipeline import Pipeline
from holisticai.metrics.bias import classification_bias_metrics, regression_bias_metrics, multiclass_bias_metrics
from tests.bias.mitigation.testing_utils.utils import (
    check_results,
    categorical_dataset,
    regression_dataset,
    multiclass_dataset
)
import pytest

warnings.filterwarnings("ignore")

seed = 42

from holisticai.mitigation.bias import MITIGATOR_NAME

def get_postprocessor(mitigator_name : MITIGATOR_NAME = "CalibratedEqualizedOdds", parameters: dict = {}):
    match mitigator_name:
        case "CalibratedEqualizedOdds":
            from holisticai.mitigation.bias import CalibratedEqualizedOdds
            return CalibratedEqualizedOdds(**parameters)
        case "EqualizedOdds":
            from holisticai.mitigation.bias import EqualizedOdds
            return EqualizedOdds(**parameters)
        case "RejectOptionClassification":
            from holisticai.mitigation.bias import RejectOptionClassification
            return RejectOptionClassification(**parameters)
        case "LPDebiaserBinary":
            from holisticai.mitigation.bias import LPDebiaserBinary
            return LPDebiaserBinary(**parameters)
        case "MLDebiaser":
            from holisticai.mitigation.bias import MLDebiaser
            return MLDebiaser(**parameters)
        case "LPDebiaserMulticlass":
            from holisticai.mitigation.bias import LPDebiaserMulticlass
            return LPDebiaserMulticlass(**parameters)
        case "PluginEstimationAndCalibration":
            from holisticai.mitigation.bias import PluginEstimationAndCalibration
            return PluginEstimationAndCalibration(**parameters)
        case "WassersteinBarycenter":
            from holisticai.mitigation.bias import WassersteinBarycenter
            return WassersteinBarycenter(**parameters)
    raise NotImplementedError


@pytest.mark.parametrize("mitigator_name, mitigator_params, fit_params", [
    ("LPDebiaserMulticlass", {}, ['y_proba','y_pred','group_a','group_b']),
    ("MLDebiaser", {}, ['y_proba','y_pred','group_a','group_b']),
])
def test_multiclass_postprocessor(mitigator_name, mitigator_params, fit_params, multiclass_dataset):
    metrics1 = run_postprocessing_categorical(multiclass_dataset, multiclass_bias_metrics, LogisticRegression, mitigator_name, fit_params, mitigator_params, is_multiclass=True)
    metrics2 = run_postprocessing_catergorical_peline(multiclass_dataset, multiclass_bias_metrics, LogisticRegression, mitigator_name, mitigator_params, is_multiclass=True)
    check_results(metrics1, metrics2)

@pytest.mark.parametrize("mitigator_name, mitigator_params, fit_params", [
    ("CalibratedEqualizedOdds", {"alpha":1.0}, ['y_proba','y_pred','group_a','group_b']),
    ("EqualizedOdds", {}, ['X','y_pred','group_a','group_b']),
    ("RejectOptionClassification", {}, ['y_proba','y_pred','group_a','group_b']),
    ("LPDebiaserBinary", {}, ['y_proba','y_pred','group_a','group_b']),
    ("MLDebiaser", {}, ['y_proba','y_pred','group_a','group_b']),
])
def test_categorical_postprocessor(mitigator_name, mitigator_params, fit_params, categorical_dataset):
    metrics1 = run_postprocessing_categorical(categorical_dataset, classification_bias_metrics, LogisticRegression, mitigator_name, fit_params, mitigator_params)
    metrics2 = run_postprocessing_catergorical_peline(categorical_dataset, classification_bias_metrics, LogisticRegression, mitigator_name, mitigator_params)
    check_results(metrics1, metrics2)


def run_postprocessing_categorical(dataset, bias_metrics, estimator_class, mitigator_name, postprocessor_fit_param_names, mitigator_params, is_multiclass=False):
    train = dataset['train']
    test = dataset['test']

    model = estimator_class(random_state=42)
    model.fit(X=train['X'], y=train['y'])
    
    post = get_postprocessor(mitigator_name, mitigator_params)
    fit_params = {'y':train['y'], 'group_a':train['group_a'], 'group_b':train['group_b'] }
    if "y_proba" in postprocessor_fit_param_names:
        y_proba = model.predict_proba(train['X'])
        fit_params.update(**{'y_proba':y_proba})
    if "y_pred" in postprocessor_fit_param_names:
        y_pred = model.predict(train['X'])
        fit_params.update(**{'y_pred':y_pred})
    post.fit(**fit_params)


    predict_params = {'y':test['y'], 'group_a':test['group_a'], 'group_b':test['group_b'] }
    if "y_proba" in postprocessor_fit_param_names:
        y_proba = model.predict_proba(test['X'])
        predict_params.update(**{'y_proba':y_proba})
    if "y_pred" in postprocessor_fit_param_names:
        y_pred = model.predict(test['X'])
        predict_params.update(**{'y_pred':y_pred})
    y_pred = post.transform(**predict_params)['y_pred']
    
    if is_multiclass:
        return bias_metrics(test['group_a'],  y_pred, test['y'])
    else:
        return bias_metrics(test['group_a'], test['group_b'], y_pred, test['y'])


def run_postprocessing_catergorical_peline(dataset, bias_metrics, estimator_class, mitigator_name, mitigator_params, is_multiclass=False):
    train = dataset['train']
    test = dataset['test']

    pipeline = Pipeline(
        steps=[
            ("estimator", estimator_class(random_state=42)),
            ("bm_postprocessing", get_postprocessor(mitigator_name, mitigator_params)),
        ]
    )

    pipeline.fit(X=train['X'], y=train['y'], bm__group_a=train['group_a'], bm__group_b=train['group_b'])
    y_pred = pipeline.predict(X=test['X'], bm__group_a=test['group_a'], bm__group_b=test['group_b'])
    
    if is_multiclass:
        return bias_metrics(test['group_a'],  y_pred, test['y'])
    else:
        return bias_metrics(test['group_a'], test['group_b'], y_pred, test['y'])

@pytest.mark.parametrize("mitigator_name, mitigator_params, fit_params", [
    ("PluginEstimationAndCalibration", {}, ['y_pred','group_a','group_b']),
    ("WassersteinBarycenter", {}, ['y_pred','group_a','group_b']),
])
def test_regression_postprocessor(mitigator_name, mitigator_params, fit_params, regression_dataset):
    metrics1 = run_postrocessing_regression(regression_dataset, LinearRegression, mitigator_name, fit_params, mitigator_params)
    metrics2 = run_postprocessing_regression_peline(regression_dataset, LinearRegression, mitigator_name, mitigator_params)
    check_results(metrics1, metrics2)

def run_postrocessing_regression(dataset, estimator_class, mitigator_name, postprocessor_fit_param_names,  mitigator_params):
    train = dataset['train']
    test = dataset['test']

    model = estimator_class()
    model_fit_parmams = {'X':train['X'], 'y':train['y']}
    model.fit(**model_fit_parmams)

    post = get_postprocessor(mitigator_name, mitigator_params)
    fit_params = {'y':train['y'], 'group_a':train['group_a'], 'group_b':train['group_b'] }
    if "y_pred" in postprocessor_fit_param_names:
        y_pred = model.predict(train['X'])
        fit_params.update(**{'y_pred':y_pred})
    post.fit(**fit_params)

    predict_params = {'y':test['y'], 'group_a':test['group_a'], 'group_b':test['group_b'] }
    y_pred = model.predict(test['X'])
    predict_params.update(**{'y_pred':y_pred})
    y_pred = post.transform(**predict_params)['y_pred']
    
    return regression_bias_metrics(test['group_a'], test['group_b'], y_pred, test['y'])

def run_postprocessing_regression_peline(dataset, estimator_class, mitigator_name, mitigator_params):
    train = dataset['train']
    test = dataset['test']

    pipeline = Pipeline(
        steps=[
            ("estimator", estimator_class()),
            ("bm_postprocessing", get_postprocessor(mitigator_name, mitigator_params)),
        ]
    )

    pipeline.fit(train['X'], train['y'], bm__group_a=train['group_a'], bm__group_b=train['group_b'])
    y_postd = pipeline.predict(test['X'], bm__group_a=test['group_a'], bm__group_b=test['group_b'])
    
    return regression_bias_metrics(test['group_a'], test['group_b'], y_postd, test['y'])
