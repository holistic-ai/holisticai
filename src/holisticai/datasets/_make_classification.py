from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.datasets import load_dataset


def make_classification():
    """Make classification dataset and train a model

    Description
    -----------
    This function loads a dataset, splits it into train and test sets,
    trains a logistic regression model and returns group membership vectors
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
        Target vector (binary)
    y_pred : array-like
        Predicted target vector (binary)

    Example
    -------
    >>> from holisticai.datasets import make_classification
    >>> data = make_classification()
    """

    dataset = load_dataset('law_school')
    dataset_split = dataset.train_test_split(test_size=0.2, random_state=42)

    # train a model, do not forget to standard scale data
    train = dataset_split['train']
    test = dataset_split['test']

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train['x'])
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, train['y'])
    X_test = scaler.transform(test['x'])
    y_pred = model.predict(X_test)

    return {
        'data':dataset['data'] if hasattr(dataset, 'data') else None,
        'output_name':dataset.__output_name__() if hasattr(dataset, '__output_name__') else None,
        'group_a':test['group_a'],
        'group_b':test['group_b'],
        'y_true':test['y'],
        'y_pred':y_pred,
    }
