import warnings
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans

from holisticai.pipeline import Pipeline
from holisticai.bias.metrics import classification_bias_metrics, regression_bias_metrics, multiclass_bias_metrics, clustering_bias_metrics
from holisticai.datasets.synthetic.recruitment import generate_rankings
from holisticai.bias.mitigation.postprocessing.debiasing_exposure.algorithm_utils import exposure_metric
from tests.bias.mitigation.utils import (
    check_results,
    categorical_dataset,
    regression_dataset,
    multiclass_dataset
)
import pytest

warnings.filterwarnings("ignore")

seed = 42

from holisticai.bias.mitigation import MITIGATOR_NAME
from holisticai.bias.mitigation import CalibratedEqualizedOdds
from holisticai.bias.mitigation import EqualizedOdds
from holisticai.bias.mitigation import RejectOptionClassification
from holisticai.bias.mitigation import LPDebiaserBinary
from holisticai.bias.mitigation import MLDebiaser
from holisticai.bias.mitigation import LPDebiaserMulticlass
from holisticai.bias.mitigation import PluginEstimationAndCalibration
from holisticai.bias.mitigation import WassersteinBarycenter
from holisticai.bias.mitigation import MCMF
from holisticai.bias.mitigation import DebiasingExposure
from holisticai.bias.mitigation import FairTopK

def get_postprocessor(mitigator_name : MITIGATOR_NAME = "CalibratedEqualizedOdds", parameters: dict = {}):
    if mitigator_name == "CalibratedEqualizedOdds":
        return CalibratedEqualizedOdds(**parameters)
    elif mitigator_name == "EqualizedOdds":
        return EqualizedOdds(**parameters)
    elif mitigator_name == "RejectOptionClassification":
        return RejectOptionClassification(**parameters)
    elif mitigator_name == "LPDebiaserBinary":
        return LPDebiaserBinary(**parameters)
    elif mitigator_name == "MLDebiaser":
        return MLDebiaser(**parameters)
    elif mitigator_name == "LPDebiaserMulticlass":
        return LPDebiaserMulticlass(**parameters)
    elif mitigator_name == "PluginEstimationAndCalibration":
        return PluginEstimationAndCalibration(**parameters)
    elif mitigator_name == "WassersteinBarycenter":
        return WassersteinBarycenter(**parameters)
    elif mitigator_name == "MCMF":
        return MCMF(**parameters)
    elif mitigator_name == "DebiasingExposure":
        return DebiasingExposure(**parameters)
    elif mitigator_name == "FairTopK":
        return FairTopK(**parameters)
    else:
        raise NotImplementedError


@pytest.mark.parametrize("mitigator_name, mitigator_params, fit_params", [
    ("LPDebiaserMulticlass", {}, ['y_proba','y_pred','group_a','group_b']),
    ("MLDebiaser", {}, ['y_proba','y_pred','group_a','group_b']),
])
def test_multiclass_postprocessor(mitigator_name, mitigator_params, fit_params, multiclass_dataset):
    metrics1 = run_postprocessing_categorical(multiclass_dataset, multiclass_bias_metrics, LogisticRegression, mitigator_name, fit_params, mitigator_params, is_multiclass=True)
    metrics2 = run_postprocessing_catergorical_pipeline(multiclass_dataset, multiclass_bias_metrics, LogisticRegression, mitigator_name, mitigator_params, is_multiclass=True)
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
    metrics2 = run_postprocessing_catergorical_pipeline(categorical_dataset, classification_bias_metrics, LogisticRegression, mitigator_name, mitigator_params)
    check_results(metrics1, metrics2)


@pytest.mark.parametrize("mitigator_name, mitigator_params, fit_params", [
    ("PluginEstimationAndCalibration", {}, ['y_pred','group_a','group_b']),
    ("WassersteinBarycenter", {}, ['y_pred','group_a','group_b']),
])
def test_regression_postprocessor(mitigator_name, mitigator_params, fit_params, regression_dataset):
    metrics1 = run_postrocessing_regression(regression_dataset, LinearRegression, mitigator_name, fit_params, mitigator_params)
    metrics2 = run_postprocessing_regression_pipeline(regression_dataset, LinearRegression, mitigator_name, mitigator_params)
    check_results(metrics1, metrics2)

@pytest.mark.parametrize("mitigator_name, mitigator_params, fit_params", [
    ("MCMF", {}, ['y_pred','group_a','group_b']),
])
def test_clustering_postprocessor(mitigator_name, mitigator_params, fit_params, regression_dataset):
    metrics1 = run_postrocessing_clustering(regression_dataset, KMeans, mitigator_name, fit_params, mitigator_params)
    metrics2 = run_postprocessing_clustering_pipeline(regression_dataset, KMeans, mitigator_name, mitigator_params)
    check_results(metrics1, metrics2)

 
def test_recsys_fairtopk():
    mitigator_name = "FairTopK"
    mitigator_params = {'top_n': 20, 'p': 0.9, 'alpha': 0.15, 'query_col':'X',
                    'score_col': 'score', 'group_col': 'protected', 'doc_col': 'Y'}
    metrics1 = run_postprocessing_fairtopk(exposure_metric, mitigator_name, mitigator_params)
    metrics2 = pd.DataFrame(columns=['Value'], data=[30.34618386900864, 0.001944], index=['exposure_ratio', 'exposure difference'])
    check_results(metrics1, metrics2)


@pytest.mark.parametrize("mitigator_name, mitigator_params", [
    ("DebiasingExposure", {'query_col': 'X', 'group_col': 'protected',
                    'score_col': 'score', 'standardize':True,
                    'gamma': 2.0, 'number_of_iterations': 10,
                    'doc_col': 'Y', 'feature_cols': ['score', 'protected'],}),
])
def test_recsys_postprocessor(mitigator_name, mitigator_params):
    import pandas as pd
    metrics1 = run_postprocessing_recsys(exposure_metric, mitigator_name, mitigator_params)
    metrics2 = pd.DataFrame(columns=['Value'], data=[0.985410, 0.001944], index=['exposure_ratio', 'exposure difference'])
    check_results(metrics1, metrics2, atol=0.1)

   
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


def run_postprocessing_catergorical_pipeline(dataset, bias_metrics, estimator_class, mitigator_name, mitigator_params, is_multiclass=False):
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


def run_postprocessing_regression_pipeline(dataset, estimator_class, mitigator_name, mitigator_params):
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


def run_postrocessing_clustering(dataset, estimator_class, mitigator_name, postprocessor_fit_param_names,  mitigator_params):
    train = dataset['train']

    model = estimator_class(random_state=42)
    model_fit_params = {'X': train['X']}
    model.fit(**model_fit_params)
    y_pred = model.predict(**model_fit_params)

    post = get_postprocessor(mitigator_name, mitigator_params)
    y_pred = post.fit_transform(train['X'], y_pred, train['group_a'], train['group_b'], model.cluster_centers_)['y_pred']
    
    return clustering_bias_metrics(train['group_a'], train['group_b'], y_pred, train['X'], model.cluster_centers_)

def run_postprocessing_clustering_pipeline(dataset, estimator_class, mitigator_name, mitigator_params):
    train = dataset['train']

    pipeline = Pipeline(
        steps=[
            ("estimator", estimator_class(random_state=42)),
            ("bm_postprocessing", get_postprocessor(mitigator_name, mitigator_params)),
        ]
    )

    pipeline.fit(train['X'])
    predict_params = {'bm__group_a':train['group_a'],
                  'bm__group_b':train['group_b'],
                  'bm__centroids':"cluster_centers_"}
    y_pred = pipeline.predict(train['X'], **predict_params)
    
    return clustering_bias_metrics(train['group_a'], train['group_b'], y_pred, train['X'], pipeline['estimator'].cluster_centers_)

def run_postprocessing_recsys(bias_metrics, mitigator_name, mitigator_params):
    rankings = generate_rankings(M=1000, k=20, p=0.25, return_p_attr=False)
    post = get_postprocessor(mitigator_name, mitigator_params)
    post.fit(rankings)
    rankings = post.transform(rankings)

    return bias_metrics(rankings, group_col='protected', query_col='X', score_col='score')


def run_postprocessing_fairtopk(bias_metrics, mitigator_name, mitigator_params):
    rankings = generate_rankings(M=1, k=20, p=0.25, return_p_attr=False)
    post = get_postprocessor(mitigator_name, mitigator_params)
    rankings = post.transform(rankings)

    return bias_metrics(rankings, group_col='protected', query_col='X', score_col='score')
