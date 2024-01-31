from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np

def append_if_not_empty(original_array, array_to_append):
    if len(array_to_append) > 0:
        original_array = np.append(original_array, array_to_append)
    return original_array

def BlackBoxAttack(attack_feature, X_train, y_train, X_test, y_test):
    """

    Description
    -----------
    The black-box attack basically trains an additional classifier (called the attack model) to predict the attacked feature's value from the remaining n-1
    features as well as the original (attacked) model's predictions.

    Parameters
    ----------

    attack_feature : str
        Target feature to be predicted
        
    X_train : pandas Dataframe
        input matrix
        
    y_train : numpy array
        Target vector of original model
        
    X_test : pandas Dataframe
        input matrix
        
    y_test : numpy array
        Target vector of original model

    Returns
    -------

    np.ndarray: Predicted output per sample.
    """
    
    categorical_features = []
    is_regression = True
    y_train_attack = X_train[attack_feature]
    X_train_attack = X_train.drop(columns = [attack_feature])
    X_test_attack = X_test.drop(columns = [attack_feature])
    X_train_attack['label'] = y_train
    X_test_attack['label'] = y_test
    if (X_train_attack.label.dtype == 'category'):
        categorical_features.append('label')
        is_regression = False
    # Create transformers for numerical and categorical features

    categorical_features = append_if_not_empty(categorical_features, X_train_attack.select_dtypes(include=['category']).columns.tolist())
    # Create transformers for numerical and categorical features
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine transformers into a preprocessor using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, X_train_attack.select_dtypes(exclude=['category']).columns),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    train_encoded = preprocessor.fit_transform(X_train_attack)
    test_encoded = preprocessor.transform(X_test_attack)

    # use half of training set for training the attack
    attack_train_ratio = 0.5
    attack_train_size = int(train_encoded.shape[0] * attack_train_ratio)
    if is_regression:
        mm = LinearRegression()
        
    else:
        mm = LogisticRegression()
    
    model = mm.fit(train_encoded[:attack_train_size], y_train_attack[:attack_train_size])
    # Predict values
    y_pred_attack = model.predict(test_encoded)
    return y_pred_attack
