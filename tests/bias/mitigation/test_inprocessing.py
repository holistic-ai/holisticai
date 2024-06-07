import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

from holisticai.pipeline import Pipeline
from holisticai.metrics.bias import classification_bias_metrics, regression_bias_metrics, multiclass_bias_metrics, recommender_bias_metrics
from tests.bias.mitigation.testing_utils.utils import (
    check_results,
    categorical_dataset,
    regression_dataset,
    multiclass_dataset,
    recommender_dataset,
)
import pytest

warnings.filterwarnings("ignore")

seed = 42

from holisticai.mitigation.bias import MITIGATOR_NAME

def get_inprocessor(mitigator_name : MITIGATOR_NAME = "CalibratedEqualizedOdds", parameters: dict = {}):
    match mitigator_name:
        case "ExponentiatedGradientReduction":
            from holisticai.mitigation.bias import ExponentiatedGradientReduction
            return ExponentiatedGradientReduction(**parameters)
        case "GridSearchReduction":
            from holisticai.mitigation.bias import GridSearchReduction
            return GridSearchReduction(**parameters)
        case "MetaFairClassifier":
            from holisticai.mitigation.bias import MetaFairClassifier
            return MetaFairClassifier(**parameters)
        case "PrejudiceRemover":
            from holisticai.mitigation.bias import PrejudiceRemover
            return PrejudiceRemover(**parameters)
        case "FairScoreClassifier":
            from holisticai.mitigation.bias import FairScoreClassifier
            return FairScoreClassifier(**parameters)
        case "PopularityPropensityMF":
            from holisticai.mitigation.bias import PopularityPropensityMF
            return PopularityPropensityMF(**parameters)
        case "BlindSpotAwareMF":
            from holisticai.mitigation.bias import BlindSpotAwareMF
            return BlindSpotAwareMF(**parameters)
        case "FairRec":
            from holisticai.mitigation.bias import FairRec
            return FairRec(**parameters)
        case "DebiasingLearningMF":
            from holisticai.mitigation.bias import DebiasingLearningMF
            return DebiasingLearningMF(**parameters)
    raise NotImplementedError


@pytest.mark.parametrize("mitigator_name, mitigator_params, model_params", [
    ("FairScoreClassifier", {'objectives':'ab', 'constraints':{}}, {"random_state":42}),
])
def test_multiclass_inprocessor(mitigator_name, mitigator_params, model_params, multiclass_dataset):
    metrics1 = run_inprocessing_categorical(multiclass_dataset, multiclass_bias_metrics, LogisticRegression, mitigator_name, model_params, mitigator_params, is_multiclass=True)
    metrics2 = run_inprocessing_catergorical_pipeline(multiclass_dataset, multiclass_bias_metrics, LogisticRegression, mitigator_name, model_params, mitigator_params, is_multiclass=True)
    check_results(metrics1, metrics2)


@pytest.mark.parametrize("mitigator_name, mitigator_params, model_params", [
    ("ExponentiatedGradientReduction", {"seed":1}, {"random_state":42}),
    ("GridSearchReduction", {},  {"random_state":42}),
    ("MetaFairClassifier", {"seed":1}, {"random_state":42}),
    ("PrejudiceRemover", {}, {"random_state":42}),
])
def test_categorical_inprocessor(mitigator_name, mitigator_params, model_params, categorical_dataset):
    metrics1 = run_inprocessing_categorical(categorical_dataset, classification_bias_metrics, LogisticRegression, mitigator_name, model_params, mitigator_params)
    metrics2 = run_inprocessing_catergorical_pipeline(categorical_dataset, classification_bias_metrics, LogisticRegression, mitigator_name, model_params, mitigator_params)
    check_results(metrics1, metrics2)


@pytest.mark.parametrize("mitigator_name, mitigator_params, model_params", [
    ("ExponentiatedGradientReduction", {'seed':1, "constraints":"BoundedGroupLoss", "loss":"Absolute", "min_val":-0.1, "max_val":1.3, "upper_bound":0.001}, {}),
    ("GridSearchReduction", {}, {}),
])
def test_regression_inprocessor(mitigator_name, mitigator_params, model_params, regression_dataset):
    metrics1 = run_inprocessing_categorical(regression_dataset, regression_bias_metrics, LinearRegression, mitigator_name, model_params, mitigator_params)
    metrics2 = run_inprocessing_catergorical_pipeline(regression_dataset, regression_bias_metrics, LinearRegression, mitigator_name, model_params, mitigator_params)
    check_results(metrics1, metrics2)


@pytest.mark.parametrize("mitigator_name, mitigator_params", [
    ("PopularityPropensityMF", {'K':40, 'beta': 0.02, 'steps':10}),
    ("BlindSpotAwareMF", {'K':40, 'beta': 0.02, 'steps':10}),
    ("FairRec", {'rec_size':10, 'MMS_fraction':0.5}),
    ("DebiasingLearningMF", {'K':40, 'normalization':'Vanilla', 'lamda':0.08, 'metric':'mse', 'bias_mode':'Regularized', 'seed':1}),
])
def test_recsys_inprocessor(mitigator_name, mitigator_params, recommender_dataset):
    metrics1 = run_inprocessing_recsys(recommender_dataset, recommender_bias_metrics, mitigator_name, mitigator_params)
    metrics2 = run_inprocessing_recsys_pipeline(recommender_dataset, recommender_bias_metrics, mitigator_name, mitigator_params)
    check_results(metrics1, metrics2)


def run_inprocessing_categorical(dataset, bias_metrics, estimator_class, mitigator_name, model_params, mitigator_params, is_multiclass=False):
    train = dataset['train']
    test = dataset['test']

    model = estimator_class(**model_params)
    
    inp = get_inprocessor(mitigator_name, mitigator_params)
    inp.transform_estimator(model)
    inp.fit(X=train['X'], y=train['y'], group_a=train['group_a'], group_b=train['group_b'])
    y_pred = inp.predict(X=test['X'], group_a=test['group_a'], group_b=test['group_b'])
    
    if is_multiclass:
        return bias_metrics(test['group_a'],  y_pred, test['y'])
    else:
        return bias_metrics(test['group_a'], test['group_b'], y_pred, test['y'])


def run_inprocessing_catergorical_pipeline(dataset, bias_metrics, estimator_class, mitigator_name, model_params, mitigator_params, is_multiclass=False):
    train = dataset['train']
    test = dataset['test']

    estimator = estimator_class(**model_params)
    inp = get_inprocessor(mitigator_name, mitigator_params)
    inp.transform_estimator(estimator)

    pipeline = Pipeline(
        steps=[
            ("bm_inprocessing", inp),
        ]
    )

    pipeline.fit(X=train['X'], y=train['y'], bm__group_a=train['group_a'], bm__group_b=train['group_b'])
    y_pred = pipeline.predict(X=test['X'], bm__group_a=test['group_a'], bm__group_b=test['group_b'])
    
    if is_multiclass:
        return bias_metrics(test['group_a'],  y_pred, test['y'])
    else:
        return bias_metrics(test['group_a'], test['group_b'], y_pred, test['y'])


def run_inprocessing_recsys(dataset, bias_metrics, mitigator_name, mitigator_params):
    train = dataset['train']
    inp = get_inprocessor(mitigator_name, mitigator_params)
    data_matrix = train['data_pivot'].fillna(0).to_numpy()
    inp.fit(data_matrix)
    rankings  = inp.predict(data_matrix, top_n=10)
    rankings = rankings.astype({'score':'float64'})
    mat = rankings.pivot(columns='Y',index='X',values='score').replace(np.nan,0).to_numpy()

    return bias_metrics(mat_pred=mat>0, metric_type='item_based')


def run_inprocessing_recsys_pipeline(dataset, bias_metrics, mitigator_name, mitigator_params):
    train = dataset['train']
    inp = get_inprocessor(mitigator_name, mitigator_params)
    data_matrix = train['data_pivot'].fillna(0).to_numpy()

    pipeline = Pipeline(
        steps=[
            ("bm_inprocessing", inp),
        ]
    )
    pipeline.fit(data_matrix)
    rankings  = pipeline.predict(data_matrix, top_n=10)
    rankings = rankings.astype({'score':'float64'})
    mat = rankings.pivot(columns='Y',index='X',values='score').replace(np.nan,0).to_numpy()

    return bias_metrics(mat_pred=mat>0, metric_type='item_based')
