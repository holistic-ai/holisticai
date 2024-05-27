import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from holisticai.datasets import load_dataset


def make_regression():
    """Make regression dataset and train a model

    Description
    -----------
    This function loads a dataset, splits it into train and test sets,
    trains a linear regression model and returns group membership vectors
    and predicted target vector.

    Parameters
    ----------
    name : str
        Name of the dataset to load

    Returns
    -------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_true : array-like
        Target vector (float)
    y_pred : array-like
        Predicted target vector (float)

    Example
    -------
    >>> from holisticai.datasets import make_regression
    >>> data = make_regression()
    """

    dataset = load_dataset('student')
    dataset_split = dataset.train_test_split(test_size=0.2, random_state=42)

    X_train = dataset_split['train']['x']  # noqa: N806
    X_test = dataset_split['test']['x']  # noqa: N806
    y_train = dataset_split['train']['y']['G3']
    y_test = dataset_split['test']['y']['G3']

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    p_attr_test = dataset_split['test']['p_attr']
    group_a = np.array(p_attr_test['sex']=='M')
    group_b = np.array(p_attr_test['sex']=='F')
    y_pred  = np.array(model.predict(X_test))
    y_true  = np.array(y_test)

    return {
        'data':dataset['data'] if hasattr(dataset, 'data') else None,
        'output_name':dataset.__output_name__() if hasattr(dataset, '__output_name__') else None,
        'group_a':group_a,
        'group_b':group_b,
        'p_attr': p_attr_test,
        'y_true':y_true,
        'y_pred':y_pred,
    }
