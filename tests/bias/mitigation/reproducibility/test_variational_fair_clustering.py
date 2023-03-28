import sys

sys.path = ["./"] + sys.path
import warnings

from sklearn.preprocessing import StandardScaler

from holisticai.bias.metrics import clustering_bias_metrics
from holisticai.bias.mitigation import VariationalFairClustering
from holisticai.pipeline import Pipeline
from tests.testing_utils._tests_utils import check_results, load_preprocessed_adult

warnings.filterwarnings("ignore")

seed = 42
train_data, test_data = load_preprocessed_adult()


def running_without_pipeline():

    X, y, group_a, group_b = train_data

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)

    model = VariationalFairClustering(nb_clusters=4, seed=seed)
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


def running_with_pipeline():
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("bm_inprocessing", VariationalFairClustering(nb_clusters=4, seed=seed)),
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


def test_reproducibility_with_and_without_pipeline():
    import numpy as np

    np.random.seed(seed)
    df1 = running_without_pipeline()
    np.random.seed(seed)
    df2 = running_with_pipeline()
    check_results(df1, df2)


test_reproducibility_with_and_without_pipeline()
