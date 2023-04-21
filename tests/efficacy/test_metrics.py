import numpy as np

def test_classification_efficacy_metrics():
    from holisticai.efficacy.metrics import classification_efficacy_metrics
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 1])
    metrics = classification_efficacy_metrics(y_true, y_pred)
    assert metrics.loc['Accuracy']['Value'] == 0.8
    
def test_regression_efficacy_metrics():
    from holisticai.efficacy.metrics import regression_efficacy_metrics
    y_pred = np.array([0.1, 0.2, 0.5, 0.2,  0.3,  0.8, 1.2, -1.2 -0.4])
    y_true = np.array([0.14, 0.2, 0.5, 0.4,  1.0,  0.8, 1.22, -3.2 -0.4])
    metrics = regression_efficacy_metrics(y_true, y_pred)
    assert np.isclose(metrics.loc['RMSE']['Value'] , 0.7526, 0.001)
    
def test_clustering_efficacy_metrics():
    from holisticai.efficacy.metrics import clustering_efficacy_metrics
    from sklearn.datasets import make_blobs
    X , y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=0.01, random_state=0)
    metrics = clustering_efficacy_metrics(X, y)
    assert np.isclose(metrics.loc['Silhouette']['Value'] , 1.0, 0.1)
    
def test_multiclassification_efficacy_metrics():
    from holisticai.efficacy.metrics import multiclassification_efficacy_metrics    
    y_pred = np.array([0, 1, 0, 2, 3, 2, 1, 4, 4, 0])
    y_true = np.array([1, 0, 3, 0, 4, 1, 3, 0, 2, 0])
    
    metrics = multiclassification_efficacy_metrics(y_true, y_pred)
    assert metrics.loc['Accuracy']['Value'] == 0.1
    
    metrics = multiclassification_efficacy_metrics(y_true, y_pred, by_class=True)
    assert np.isclose(metrics.loc['F1-Score'][0], 0.2857, 0.001)
    
    y_pred = np.array(["Low", "Medium", "High", "Low", "High", "Low", "Low", "Medium", "Medium", "High"])
    y_true = np.array(["Low", "High", "High", "Low", "Medium", "Low", "Low", "Medium", "Medium", "High"])
    
    metrics = multiclassification_efficacy_metrics(y_true, y_pred, classes=["Low","Medium","High"], by_class=True)
    assert np.isclose(metrics.loc['F1-Score'][0], 0.8, 0.001)
    