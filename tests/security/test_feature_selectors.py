from holisticai.datasets import load_dataset
from holisticai.security.commons import DataMinimizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from holisticai.utils import BinaryClassificationProxy

dataset = load_dataset('adult')

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

y_pred = model.predict(test['X'])
y_proba = model.predict_proba(test['X'])

print('Base model accuracy: ', model.score(test['X'], test['y']))

proxy = BinaryClassificationProxy(predict=model.predict,
                                    predict_proba=model.predict_proba,
                                    classes=[0, 1])

entry = DataMinimizer(proxy=proxy)

entry.fit(train['X'], train['y'])