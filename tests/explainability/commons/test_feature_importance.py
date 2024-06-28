import pandas as pd
from holisticai.explainability.commons._feature_importance import filter_feature_importance

def test_filter_feature_importance():
    # Create a sample feature importance dataframe
    feature_importance = pd.DataFrame({
        'Variable': ['feature1', 'feature2', 'feature3', 'feature4'],
        'Importance': [0.3, 0.2, 0.1, 0.4]
    })

    # Test filtering with default threshold (alpha=None)
    filtered_df = filter_feature_importance(feature_importance)
    filtered_df.reset_index(drop=True, inplace=True)
    expected_filtered_df = pd.DataFrame({
        'Variable': ['feature4', 'feature1'],
        'Importance': [0.4, 0.3]
    })
    assert filtered_df.equals(expected_filtered_df)

    # Test filtering with custom threshold (alpha=0.8)
    filtered_df = filter_feature_importance(feature_importance, alpha=0.8)
    filtered_df.reset_index(drop=True, inplace=True)
    expected_filtered_df = pd.DataFrame({
        'Variable': ['feature4', 'feature1'],
        'Importance': [0.4, 0.3]
    })
    assert filtered_df.equals(expected_filtered_df)

    # Test filtering with custom threshold (alpha=0.5)
    filtered_df = filter_feature_importance(feature_importance, alpha=0.5)
    filtered_df.reset_index(drop=True, inplace=True)
    expected_filtered_df = pd.DataFrame({
        'Variable': ['feature4'],
        'Importance': [0.4]
    })
    assert filtered_df.equals(expected_filtered_df)

    # Test filtering with custom threshold (alpha=0.2)
    filtered_df = filter_feature_importance(feature_importance, alpha=0.2)
    filtered_df.reset_index(drop=True, inplace=True)
    expected_filtered_df = pd.DataFrame({
        'Variable': ['feature4'],
        'Importance': [0.4]
    })
    assert filtered_df.equals(expected_filtered_df)