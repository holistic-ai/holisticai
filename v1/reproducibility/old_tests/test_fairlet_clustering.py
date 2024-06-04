import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import warnings

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from holisticai.metrics.bias import clustering_bias_metrics
from holisticai.mitigation.bias import FairletClustering, FairletClusteringPreprocessing
from holisticai.pipeline import Pipeline
from tests.bias.mitigation.testing_utils.utils import (
    check_results,
    small_clustering_dataset as ds,
)

warnings.filterwarnings("ignore")

seed = 42


def running_without_pipeline(ds):
    train = ds['train']
    test = ds['test']

    scaler = StandardScaler()
    Xt = scaler.fit_transform(train['x'])

    model = FairletClustering(n_clusters=4, seed=seed)
    model.fit(Xt, group_a=train['group_a'], group_b=train['group_b'])

    # Test
    Xt = scaler.transform(test['x'])

    y_pred = model.predict(Xt, group_a=test['group_a'], group_b=test['group_b'])
    centroids = model.cluster_centers_
    df = clustering_bias_metrics(
        test['group_a'], test['group_b'], y_pred, centroids=centroids, metric_type="both"
    )
    return df


def running_with_pipeline(ds):
    train = ds['train']
    test = ds['test']

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("bm_inprocessing", FairletClustering(n_clusters=4, seed=seed)),
        ]
    )

    fit_params = {"bm__group_a": train['group_a'], "bm__group_b": train['group_b']}

    pipeline.fit(train['x'], train['y'], **fit_params)

    predict_params = {
        "bm__group_a": train['group_a'],
        "bm__group_b": train['group_b'],
    }
    y_pred = pipeline.predict(test['x'], **predict_params)
    centroids = pipeline["bm_inprocessing"].cluster_centers_
    df = clustering_bias_metrics(
        test['group_a'], test['group_b'], y_pred, centroids=centroids, metric_type="both"
    )
    return df


def running_without_pipeline_pre(ds):
    
    train = ds['train']
    test = ds['test']

    scaler = StandardScaler()
    Xt = scaler.fit_transform(train['x'])

    prep = FairletClusteringPreprocessing(p=1, q=3)
    Xt = prep.fit_transform(Xt, group_a=train['group_a'], group_b=train['group_b'])
    sample_weight = prep.estimator_params["sample_weight"]

    model = KMeans(n_clusters=4)
    model.fit(Xt, sample_weight=sample_weight)

    # Test
    Xt = scaler.transform(test['x'])
    Xt = prep.transform(Xt, group_a=test['group_a'], group_b=test['group_b'])
    y_pred = model.predict(Xt)
    centroids = model.cluster_centers_
    df = clustering_bias_metrics(
        test['group_a'], test['group_b'], y_pred, centroids=centroids, metric_type="both"
    )
    return df


def running_with_pipeline_pre(ds):
    train = ds['train']
    test = ds['test']

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("bm_preprocessing", FairletClusteringPreprocessing(p=1, q=3)),
            ("cluster", KMeans(n_clusters=4)),
        ]
    )

    fit_params = {"bm__group_a": train['group_a'], "bm__group_b": train['group_b']}

    pipeline.fit(train['x'], train['y'], **fit_params)

    predict_params = {
        "bm__group_a": train['group_a'],
        "bm__group_b": train['group_b'],
    }
    y_pred = pipeline.predict(test['x'], **predict_params)
    centroids = pipeline["cluster"].cluster_centers_
    df = clustering_bias_metrics(
        test['group_a'], test['group_b'], y_pred, centroids=centroids, metric_type="both"
    )
    return df


def test_reproducibility_with_and_without_pipeline(ds):
    np.random.seed(seed)
    df1 = running_without_pipeline(ds)
    np.random.seed(seed)
    df2 = running_with_pipeline(ds)
    check_results(df1, df2)


def test_reproducibility_with_and_without_pipeline_pre(ds):
    np.random.seed(seed)
    df1 = running_without_pipeline_pre(ds)
    np.random.seed(seed)
    df2 = running_with_pipeline_pre(ds)
    check_results(df1, df2)
