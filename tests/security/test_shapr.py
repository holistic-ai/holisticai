from holisticai.security.metrics import shapr_score
from holisticai.datasets import load_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tests.security.utils import categorical_dataset

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

def test_shapr(categorical_dataset):
    train = categorical_dataset['train']
    test = categorical_dataset['test']

    preprocessor = create_preprocessor(train['X'])
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', DecisionTreeClassifier())])

    model.fit(train['X'], train['y'])

    pred_train = model.predict(train['X'])
    pred_test = model.predict(test['X'])
    v = shapr_score(train['y'], test['y'], pred_train, pred_test) 
    print(v)
