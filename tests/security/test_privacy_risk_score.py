import numpy as np
from holisticai.security.metrics import privacy_risk_score

def test_privacy_risk_score():
    # Generate synthetic data for testing
    shadow_train_probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.9, 0.1]])
    shadow_train_labels = np.array([0, 1, 0])
    shadow_test_probs = np.array([[0.7, 0.3], [0.5, 0.5], [0.4, 0.6]])
    shadow_test_labels = np.array([0, 1, 1])
    target_train_probs = np.array([[0.85, 0.15], [0.65, 0.35], [0.75, 0.25]])
    target_train_labels = np.array([0, 1, 0])

    # Expected output is not known, so we will just check the shape and type
    risk_scores = privacy_risk_score(
        (shadow_train_probs, shadow_train_labels),
        (shadow_test_probs, shadow_test_labels),
        (target_train_probs, target_train_labels)
    )

    assert isinstance(risk_scores, np.ndarray), "Output should be a numpy array"
    assert risk_scores.shape == (3,), "Output shape should match the number of target training samples"
