import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from testing_utils.tests_utils import small_clustering_dataset

from holisticai.bias.metrics import clustering_bias_metrics
from holisticai.bias.mitigation.postprocessing.mcmf_clustering.transformer import MCMF
from holisticai.pipeline import Pipeline


def test_using_pipeline(small_clustering_dataset):
    np.random.seed(100)
    k = 4
    X_train, _, group_a_train, group_b_train = [
        d[:1000] for d in small_clustering_dataset[0]
    ]

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", KMeans(n_clusters=k)),
            ("bm_postprocessing", MCMF(metric="L1", group_mode="ab", verbose=1)),
        ]
    )

    pipeline.fit(X_train)
    predict_params = {
        "bm__group_a": group_a_train,
        "bm__group_b": group_b_train,
        "bm__centroids": "cluster_centers_",
    }
    y_pred = pipeline.predict(X_train, **predict_params)
    p_attr = np.array(group_a_train).reshape(-1)
    items_per_cluster = [len(np.where((y_pred == i) & p_attr)[0]) for i in range(k)]
    assert np.abs(np.max(items_per_cluster) - np.min(items_per_cluster)) <= 1


def test_withoutpipeline(small_clustering_dataset):
    np.random.seed(100)
    k = 4
    X_train, _, group_a_train, group_b_train = [
        d[:1000] for d in small_clustering_dataset[0]
    ]
    Xt = StandardScaler().fit_transform(X_train)
    model = KMeans(n_clusters=k)
    model.fit(Xt)
    y_pred = model.predict(Xt)
    pos = MCMF(metric="L1", verbose=1)
    prediction = pos.fit_transform(
        Xt, y_pred, group_a_train, group_b_train, centroids=model.cluster_centers_
    )
    new_y_pred = prediction["y_pred"]
    metric = clustering_bias_metrics(
        group_a_train,
        group_b_train,
        new_y_pred,
        data=Xt,
        centroids=model.cluster_centers_,
        metric_type="both",
    )
    p_attr = np.array(group_a_train).reshape(-1)
    assert np.sum(new_y_pred == y_pred) / len(y_pred) > 0.5
    items_per_cluster = [len(np.where((new_y_pred == i) & p_attr)[0]) for i in range(k)]
    assert np.abs(np.max(items_per_cluster) - np.min(items_per_cluster)) <= 1
