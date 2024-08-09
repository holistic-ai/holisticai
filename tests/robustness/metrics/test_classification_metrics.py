import numpy as np
from holisticai.robustness.metrics import adversarial_accuracy, empirical_robustness

def test_adversarial_accuracy():
    y = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    y_adv_pred = np.array([0, 1, 0, 0, 0])
    
    accuracy = adversarial_accuracy(y, y_pred, y_adv_pred)
    assert accuracy == 0.75
    
    y = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0])
    y_adv_pred = np.array([0, 1, 0, 1, 0])
    
    accuracy = adversarial_accuracy(y, y_pred, y_adv_pred)
    assert accuracy == 1.0

    y = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0])
    y_adv_pred = np.array([1, 0, 1, 0, 1])
    
    accuracy = adversarial_accuracy(y, y_pred, y_adv_pred)
    assert accuracy == 0.0


def test_empirical_robustness():
    x = np.array([[1, 1], [2, 2]])
    adv_x = np.array([[1.5, 1.5], [3, 3]])
    y_pred = np.array([0, 1])
    y_adv_pred = np.array([1, 0])

    robustness = empirical_robustness(x, adv_x, y_pred, y_adv_pred, norm=2)
    assert np.isclose(robustness, 0.5)
    
    x = np.array([[1, 2, 3], [4, 5, 6]])
    adv_x = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    y_pred = np.array([0, 0])
    y_adv_pred = np.array([1, 1])
    
    robustness = empirical_robustness(x, adv_x, y_pred, y_adv_pred, norm=2)
    assert np.isclose(robustness, 0.1)
    
    x = np.array([[1, 2, 3], [4, 5, 6]])
    adv_x = np.array([[1, 2, 3], [4, 5, 6]])
    y_pred = np.array([0, 1])
    y_adv_pred = np.array([0, 1])
    
    robustness = empirical_robustness(x, adv_x, y_pred, y_adv_pred, norm=2)
    assert np.isclose(robustness, 0.0)
