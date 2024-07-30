import pytest
import numpy as np

def train_estimator(train, learning_task):
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression, LogisticRegression

    # Identify categorical and numerical features
    categorical_features = train['X'].select_dtypes(include=['category']).columns
    numerical_fatures = train['X'].select_dtypes(exclude=['category']).columns

    # Create transformers for numerical and categorical features
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine transformers into a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_fatures),
            ('cat', categorical_transformer, categorical_features)
    ])

    if learning_task=='regression':
        from holisticai.utils import RegressionProxy
        model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LinearRegression())])
        model.fit(train['X'], train['y'])
        proxy = RegressionProxy(predict=model.predict)

    elif learning_task=='multi_classification':
        from holisticai.utils import MultiClassificationProxy
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', LogisticRegression())])
        model.fit(train['X'], train['y'])
        proxy = MultiClassificationProxy(predict=model.predict,
                                        predict_proba=model.predict_proba,
                                        classes=model.classes_)
        
    elif learning_task=='binary_classification':
        from holisticai.utils import BinaryClassificationProxy
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', LogisticRegression())])
        model.fit(train['X'], train['y'])
        proxy = BinaryClassificationProxy(predict=model.predict,
                                        predict_proba=model.predict_proba,
                                        classes=model.classes_)

    
    return proxy

@pytest.mark.parametrize("learning_task, reference_score, dataset_name", [
    ("binary_classification", 1.0006506180871828, 'adult'),
    ("multi_classification", 1.0, 'student_multiclass'),
    ("regression", 0.04, 'us_crime'),
])
def test_data_minimization(learning_task, reference_score, dataset_name):
    from holisticai.datasets import load_dataset
    
    from holisticai.security.commons import DataMinimizer
    from holisticai.security.metrics import data_minimization_score

    dataset = load_dataset(dataset_name, preprocessed=True)
    train_test = dataset.train_test_split(0.2, random_state=42)
    train = train_test['train']
    test = train_test['test']

    proxy = train_estimator(train, learning_task)

    entry = DataMinimizer(proxy=proxy)

    entry.fit(train['X'], train['y'])
    y_pred_dm = entry.predict(test['X'])
    y_pred = proxy.predict(test['X'])

    metric = data_minimization_score(test['y'], y_pred, y_pred_dm)
    assert np.isclose(metric, reference_score, atol=0.1)