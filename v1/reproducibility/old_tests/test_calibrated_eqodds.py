import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.metrics.bias import classification_bias_metrics
from holisticai.mitigation.bias import CalibratedEqualizedOdds
from holisticai.pipeline import Pipeline
from tests.bias.mitigation.testing_utils.utils import (
    check_results,
    small_categorical_dataset,
)

warnings.filterwarnings("ignore")

seed = 42


def running_without_pipeline(small_categorical_dataset):

    train = small_categorical_dataset['train']
    test = small_categorical_dataset['test']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train['x'])

    model = LogisticRegression()
    model.fit(X_train_scaled, train['y'])

    y_train_proba = model.predict_proba(X_train_scaled)

    post = CalibratedEqualizedOdds("fpr")
    post.fit(train['y'], y_train_proba, group_a=train['group_a'], group_b=train['group_b'])

    # Test
    X_test_scaled = scaler.transform(test['x'])

    y_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)
    y_pred = post.transform(
        y_pred, y_test_proba, group_a=test['group_a'], group_b=test['group_b']
    )["y_pred"]
    
    return classification_bias_metrics(test['group_a'], test['group_b'], y_pred, test['y'])


def running_with_pipeline(small_categorical_dataset):
    train = small_categorical_dataset['train']
    test = small_categorical_dataset['test']


    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("estimator", LogisticRegression()),
            ("bm_posprocessing", CalibratedEqualizedOdds("fpr")),
        ]
    )

    pipeline.fit(train['x'], train['y'], bm__group_a=train['group_a'], bm__group_b=train['group_b'])
    y_pred = pipeline.predict(test['x'], bm__group_a=test['group_a'], bm__group_b=test['group_b'])
    
    return classification_bias_metrics(test['group_a'], test['group_b'], y_pred, test['y'])


def test_reproducibility_with_and_without_pipeline(small_categorical_dataset):
    import numpy as np

    np.random.seed(seed)
    df1 = running_without_pipeline(small_categorical_dataset)
    np.random.seed(seed)
    df2 = running_with_pipeline(small_categorical_dataset)
    check_results(df1, df2)
