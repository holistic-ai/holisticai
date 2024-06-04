import warnings

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

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

def get_preprocessor(mitigator_name : MITIGATOR_NAME = "CorrelationRemover", parameters: dict = {}):
    match mitigator_name:
        case "CorrelationRemover":
            from holisticai.mitigation.bias import CorrelationRemover
            return CorrelationRemover(**parameters)
        case "Reweighing":
            from holisticai.mitigation.bias import Reweighing
            return Reweighing(**parameters)
        case "LearningFairRepresentation":
            from holisticai.mitigation.bias import LearningFairRepresentation
            return LearningFairRepresentation(**parameters)
    raise NotImplementedError

@pytest.mark.parametrize("mitigator_name, mitigator_params, fit_params, extra_model_fit_params", [
    ("CorrelationRemover", {"alpha":1.0}, ['X','group_a','group_b'], []),
    ("Reweighing", {}, ['X','y','group_a','group_b'], ['sample_weight']),
])
def test_multiclass_preprocessor(mitigator_name, mitigator_params, fit_params, extra_model_fit_params, multiclass_dataset):
    metrics1 = run_preprocessing_categorical(multiclass_dataset, multiclass_bias_metrics, LogisticRegression, mitigator_name, fit_params, extra_model_fit_params, mitigator_params, is_multiclass=True)
    metrics2 = run_preprocessing_catergorical_pipeline(multiclass_dataset, multiclass_bias_metrics, LogisticRegression, mitigator_name, mitigator_params, is_multiclass=True)
    check_results(metrics1, metrics2)

@pytest.mark.parametrize("mitigator_name, mitigator_params, fit_params, extra_model_fit_params", [
    ("CorrelationRemover", {"alpha":1.0}, ['X','group_a','group_b'], []),
    ("Reweighing", {}, ['X','y','group_a','group_b'], ['sample_weight']),
    ("LearningFairRepresentation", {}, ['X','y','group_a','group_b'], [])
])
def test_categorical_preprocessor(mitigator_name, mitigator_params, fit_params, extra_model_fit_params, categorical_dataset):
    metrics1 = run_preprocessing_categorical(categorical_dataset, classification_bias_metrics, LogisticRegression, mitigator_name, fit_params, extra_model_fit_params, mitigator_params)
    metrics2 = run_preprocessing_catergorical_pipeline(categorical_dataset, classification_bias_metrics, LogisticRegression, mitigator_name, mitigator_params)
    check_results(metrics1, metrics2)

@pytest.mark.parametrize("mitigator_name, mitigator_params, fit_params", [
    ("CorrelationRemover", {"alpha":1.0}, ['X','group_a','group_b']),
])
def test_regression_preprocessor(mitigator_name, mitigator_params, fit_params, regression_dataset):
    metrics1 = run_preprocessing_regression(regression_dataset, LinearRegression, mitigator_name, fit_params, mitigator_params)
    metrics2 = run_preprocessing_regression_pipeline(regression_dataset, LinearRegression, mitigator_name, mitigator_params)
    check_results(metrics1, metrics2)


def run_preprocessing_categorical(dataset, bias_metrics, estimator_class, mitigator_name, preprocessor_fit_param_names, extra_model_fit_params,  mitigator_params, is_multiclass=False):
    train = dataset['train']
    test = dataset['test']

    pre = get_preprocessor(mitigator_name, mitigator_params)
    xt = pre.fit_transform(**{p:train[p] for p in preprocessor_fit_param_names})

    model = estimator_class(random_state=42)
    model_fit_parmams = {'X':xt, 'y':train['y']}
    
    if 'sample_weight' in extra_model_fit_params:
        model_fit_parmams.update({'sample_weight': pre.sample_weight})
    model.fit(**model_fit_parmams)

    xt = pre.transform(test['X'], group_a=test['group_a'], group_b=test['group_b'])
    y_pred = model.predict(xt)   
    
    if is_multiclass:
        return bias_metrics(test['group_a'],  y_pred, test['y'])
    else:
        return bias_metrics(test['group_a'], test['group_b'], y_pred, test['y'])

def run_preprocessing_catergorical_pipeline(dataset, bias_metrics, estimator_class, mitigator_name, mitigator_params, is_multiclass=False):
    train = dataset['train']
    test = dataset['test']

    pipeline = Pipeline(
        steps=[
            ("bm_preprocessing", get_preprocessor(mitigator_name, mitigator_params)),
            ("estimator", estimator_class(random_state=42)),
        ]
    )

    pipeline.fit(train['X'], train['y'], bm__group_a=train['group_a'], bm__group_b=train['group_b'])
    y_pred = pipeline.predict(test['X'], bm__group_a=test['group_a'], bm__group_b=test['group_b'])
    
    if is_multiclass:
        return bias_metrics(test['group_a'],  y_pred, test['y'])
    else:
        return bias_metrics(test['group_a'], test['group_b'], y_pred, test['y'])
    
def run_preprocessing_regression(dataset, estimator_class, mitigator_name, preprocessor_fit_param_names,  mitigator_params):
    train = dataset['train']
    test = dataset['test']

    pre = get_preprocessor(mitigator_name, mitigator_params)
    xt = pre.fit_transform(**{p:train[p] for p in preprocessor_fit_param_names})

    model = estimator_class()
    model_fit_parmams = {'X':xt, 'y':train['y']}
    model.fit(**model_fit_parmams)

    xt = pre.transform(test['X'], group_a=test['group_a'], group_b=test['group_b'])
    y_pred = model.predict(xt)   
    
    return regression_bias_metrics(test['group_a'], test['group_b'], y_pred, test['y'])

def run_preprocessing_regression_pipeline(dataset, estimator_class, mitigator_name, mitigator_params):
    train = dataset['train']
    test = dataset['test']

    pipeline = Pipeline(
        steps=[
            ("bm_preprocessing", get_preprocessor(mitigator_name, mitigator_params)),
            ("estimator", estimator_class()),
        ]
    )

    pipeline.fit(train['X'], train['y'], bm__group_a=train['group_a'], bm__group_b=train['group_b'])
    y_pred = pipeline.predict(test['X'], bm__group_a=test['group_a'], bm__group_b=test['group_b'])
    
    return regression_bias_metrics(test['group_a'], test['group_b'], y_pred, test['y'])