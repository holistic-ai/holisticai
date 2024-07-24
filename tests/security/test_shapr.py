from holisticai.security.metrics import shapr_score
from holisticai.datasets import load_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_preprocessor(X):
    categorical_features = X.select_dtypes(include=['category']).columns
    numerical_fatures = X.select_dtypes(exclude=['category']).columns

    # Create transformers for numerical and categorical features
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine transformers into a preprocessor using ColumnTransformer
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_fatures),
            ('cat', categorical_transformer, categorical_features)
    ])

def test_shapr():
    dataset = load_dataset('adult', preprocessed=True)
    train_test = dataset.train_test_split(0.2, random_state=42)
    train = train_test['train']
    test = train_test['test']

    preprocessor = create_preprocessor(train['X'])
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', DecisionTreeClassifier())])

    model.fit(train['X'], train['y'])

    pred_train = model.predict(train['X'])
    pred_test = model.predict(test['X'])
    v = shapr_score(train['y'], test['y'], pred_train, pred_test) 
    print(v)

def test_data_minimization():
    from holisticai.datasets import load_dataset
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    from holisticai.utils import BinaryClassificationProxy
    from holisticai.security.commons import DataMinimizer
    from holisticai.security.metrics import data_minimization_accuracy_ratio

    dataset = load_dataset('adult', preprocessed=False)
    train_test = dataset.train_test_split(0.2, random_state=42)
    train = train_test['train']
    test = train_test['test']

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
    
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', DecisionTreeClassifier())])

    model.fit(train['X'], train['y'])

    proxy = BinaryClassificationProxy(predict=model.predict,
                                            predict_proba=model.predict_proba,
                                            classes=[0, 1])

    entry = DataMinimizer(proxy=proxy)

    entry.fit(train['X'], train['y'])
    y_pred_dm = entry.predict(test['X'])
    y_pred = proxy.predict(test['X'])

    metric = data_minimization_accuracy_ratio(test['y'], y_pred, y_pred_dm)
    print(metric)