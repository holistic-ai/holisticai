import warnings

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from holisticai.pipeline import Pipeline
from holisticai.bias.metrics import classification_bias_metrics, regression_bias_metrics, multiclass_bias_metrics, clustering_bias_metrics
from holisticai.datasets.synthetic.recruitment import generate_rankings
from holisticai.bias.mitigation.postprocessing.debiasing_exposure.algorithm_utils import exposure_metric

from tests.bias.mitigation.utils import (
    check_results,
    categorical_dataset,
    regression_dataset,
    multiclass_dataset,
    clustering_dataset
)
import pytest

warnings.filterwarnings("ignore")

seed = 42

from holisticai.bias.mitigation import MITIGATOR_NAME
from holisticai.bias.mitigation import CorrelationRemover
from holisticai.bias.mitigation import Reweighing
from holisticai.bias.mitigation import LearningFairRepresentation
from holisticai.bias.mitigation import FairletClusteringPreprocessing
from holisticai.bias.mitigation import DisparateImpactRemoverRS

def get_preprocessor(mitigator_name : MITIGATOR_NAME = "CorrelationRemover", parameters: dict = {}):
    if mitigator_name == "CorrelationRemover":
        return CorrelationRemover(**parameters)
    elif mitigator_name == "Reweighing":
        return Reweighing(**parameters)
    elif mitigator_name == "LearningFairRepresentation":
        return LearningFairRepresentation(**parameters)
    elif mitigator_name == "FairletClusteringPreprocessing":
        return FairletClusteringPreprocessing(**parameters)
    elif mitigator_name == "DisparateImpactRemoverRS":
        return DisparateImpactRemoverRS(**parameters)
    else:
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

@pytest.mark.parametrize("mitigator_name, mitigator_params, fit_params", [
    ("FairletClusteringPreprocessing", {"seed":42, "p":1, "q":3}, ['X','group_a','group_b']),
])
def test_clustering_preprocessor(mitigator_name, mitigator_params, fit_params, clustering_dataset):
    metrics1 = run_preprocessing_clustering(clustering_dataset, KMeans, mitigator_name, fit_params, mitigator_params)
    metrics2 = run_preprocessing_clustering_pipeline(clustering_dataset, KMeans, mitigator_name, mitigator_params)
    check_results(metrics1, metrics2)

@pytest.mark.parametrize("mitigator_name, mitigator_params", [
    ("DisparateImpactRemoverRS", {'query_col': 'X', 'group_col': 'protected', 'score_col': 'score', 'repair_level':1}),
])
def test_recsys_preprocessor(mitigator_name, mitigator_params):
    import pandas as pd
    metrics1 = run_preprocessing_recsys(exposure_metric, mitigator_name, mitigator_params)
    metrics2 = pd.DataFrame(columns=['Value'], data=[1.0037613239703396, 0.001944], index=['exposure_ratio', 'exposure difference'])
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

def run_preprocessing_clustering(dataset, estimator_class, mitigator_name, fit_params_name, mitigator_params):
    train = dataset['train']

    pre = get_preprocessor(mitigator_name, mitigator_params)
    fit_params = {p:train[p] for p in fit_params_name}
    xt = pre.fit_transform(**fit_params)

    model = estimator_class(n_clusters=4, random_state=42)
    model.fit(xt)

    y_pred = model.predict(xt)
    centroids = model.cluster_centers_
    return clustering_bias_metrics(train['group_a'], train['group_b'], y_pred, train['X'], centroids)

def run_preprocessing_clustering_pipeline(dataset, estimator_class, mitigator_name, mitigator_params):
    train = dataset['train']

    pipeline = Pipeline(
        steps=[
            ("bm_preprocessing", get_preprocessor(mitigator_name, mitigator_params)),
            ("estimator", estimator_class(n_clusters=4, random_state=42)),
        ]
    )

    fit_params_dict = {
        "bm__group_a": train['group_a'], 
        "bm__group_b": train['group_b']
        }
    
    pipeline.fit(train['X'], **fit_params_dict)

    y_pred = pipeline.predict(train['X'])
    centroids = pipeline.named_steps['estimator'].cluster_centers_

    return clustering_bias_metrics(train['group_a'], train['group_b'], y_pred, train['X'], centroids)

def run_preprocessing_recsys(bias_metrics, mitigator_name, mitigator_params):
    rankings = generate_rankings(M=1000, k=20, p=0.25, return_p_attr=False)
    pre = get_preprocessor(mitigator_name, mitigator_params)
    rankings = pre.transform(rankings)

    return bias_metrics(rankings, group_col='protected', query_col='X', score_col='score')
