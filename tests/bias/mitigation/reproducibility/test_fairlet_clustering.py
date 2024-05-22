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
    small_clustering_dataset,
)

warnings.filterwarnings("ignore")

seed = 42


def running_without_pipeline(small_clustering_dataset):
    train_data, test_data = small_clustering_dataset
    X, y, group_a, group_b = train_data

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    model = FairletClustering(n_clusters=4, seed=seed)
    model.fit(Xt, group_a=group_a, group_b=group_b)

    # Test
    X, y, group_a, group_b = test_data
    Xt = scaler.transform(X)

    y_pred = model.predict(Xt, group_a=group_a, group_b=group_b)
    centroids = model.cluster_centers_
    df = clustering_bias_metrics(
        group_a, group_b, y_pred, centroids=centroids, metric_type="both"
    )
    return df


def running_with_pipeline(small_clustering_dataset):
    train_data, test_data = small_clustering_dataset
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("bm_inprocessing", FairletClustering(n_clusters=4, seed=seed)),
        ]
    )

    X, y, group_a, group_b = train_data
    fit_params = {"bm__group_a": group_a, "bm__group_b": group_b}

    pipeline.fit(X, y, **fit_params)

    X, y, group_a, group_b = test_data
    predict_params = {
        "bm__group_a": group_a,
        "bm__group_b": group_b,
    }
    y_pred = pipeline.predict(X, **predict_params)
    centroids = pipeline["bm_inprocessing"].cluster_centers_
    df = clustering_bias_metrics(
        group_a, group_b, y_pred, centroids=centroids, metric_type="both"
    )
    return df


def running_without_pipeline_pre(small_clustering_dataset):
    train_data, test_data = small_clustering_dataset
    X, y, group_a, group_b = train_data

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    prep = FairletClusteringPreprocessing(p=1, q=3)
    Xt = prep.fit_transform(Xt, group_a=group_a, group_b=group_b)
    sample_weight = prep.estimator_params["sample_weight"]

    model = KMeans(n_clusters=4)
    model.fit(Xt, sample_weight=sample_weight)

    # Test
    X, y, group_a, group_b = test_data
    Xt = scaler.transform(X)
    Xt = prep.transform(Xt, group_a=group_a, group_b=group_b)
    y_pred = model.predict(Xt)
    centroids = model.cluster_centers_
    df = clustering_bias_metrics(
        group_a, group_b, y_pred, centroids=centroids, metric_type="both"
    )
    return df


def running_with_pipeline_pre(small_clustering_dataset):
    train_data, test_data = small_clustering_dataset
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("bm_preprocessing", FairletClusteringPreprocessing(p=1, q=3)),
            ("cluster", KMeans(n_clusters=4)),
        ]
    )

    X, y, group_a, group_b = train_data
    fit_params = {"bm__group_a": group_a, "bm__group_b": group_b}

    pipeline.fit(X, y, **fit_params)

    X, y, group_a, group_b = test_data
    predict_params = {
        "bm__group_a": group_a,
        "bm__group_b": group_b,
    }
    y_pred = pipeline.predict(X, **predict_params)
    centroids = pipeline["cluster"].cluster_centers_
    df = clustering_bias_metrics(
        group_a, group_b, y_pred, centroids=centroids, metric_type="both"
    )
    return df


def test_reproducibility_with_and_without_pipeline(small_clustering_dataset):
    np.random.seed(seed)
    df1 = running_without_pipeline(small_clustering_dataset)
    np.random.seed(seed)
    df2 = running_with_pipeline(small_clustering_dataset)
    check_results(df1, df2)


def test_reproducibility_with_and_without_pipeline_pre(small_clustering_dataset):
    np.random.seed(seed)
    df1 = running_without_pipeline_pre(small_clustering_dataset)
    np.random.seed(seed)
    df2 = running_with_pipeline_pre(small_clustering_dataset)
    check_results(df1, df2)
