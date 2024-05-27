import numpy as np
from sklearn.ensemble import RandomForestClassifier

from holisticai.datasets import load_dataset


def make_multiclass():
    """Make multiclassification dataset and train a model

    Description
    -----------
    This function loads a dataset, splits it into train and test sets,
    trains a random forest model and returns group membership vectors
    and predicted target vector.

    Parameters
    ----------
    None

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
    >>> from holisticai.datasets import make_multiclassification
    >>> data = make_multiclassification()
    """

    dataset = load_dataset('student_multiclass')
    dataset = dataset.map(lambda x: {'p_attr': x['p_attr']['sex']})
    dataset_split = dataset.train_test_split(test_size=0.2, random_state=42)

    # Train a simple Random Forest Classifier
    x_train = dataset_split['train']['x']
    y_train = dataset_split['train']['y']

    model = RandomForestClassifier(random_state=111)
    model.fit(x_train, y_train)

    # Predict values
    x_test = dataset_split['test']['x']
    y_test = dataset_split['test']['y']
    p_attr_test = dataset_split['test']['p_attr']
    group_a = np.array(p_attr_test=='M')
    group_b = np.array(p_attr_test=='F')
    y_pred = model.predict(x_test)

    return {
        'data':dataset['data'] if hasattr(dataset, 'data') else None,
        'output_name':dataset.__output_name__() if hasattr(dataset, '__output_name__') else None,
        'group_a':group_a,
        'group_b':group_b,
        'p_attr': p_attr_test,
        'y_true':y_test,
        'y_pred':y_pred,
    }
