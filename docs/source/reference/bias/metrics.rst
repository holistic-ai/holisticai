

:py:mod:`holisticai.bias.metrics`
=================================

.. automodule:: holisticai.bias.metrics
    :no-members:
    :no-inherited-members:

**Binary Classification**

.. autosummary::
   :nosignatures:
   :template: function.rst
   :toctree: .generated/

    abroca
    accuracy_diff
    average_odds_diff
    classification_bias_metrics
    cohen_d
    disparate_impact
    equal_opportunity_diff
    false_negative_rate_diff
    false_positive_rate_diff
    four_fifths
    statistical_parity
    true_negative_rate_diff
    z_test_diff
    z_test_ratio

**Multiclass Classification**

.. autosummary::
    :nosignatures:
    :template: function.rst
    :toctree: .generated/

    accuracy_matrix
    confusion_matrix
    confusion_tensor
    frequency_matrix
    multiclass_average_odds
    multiclass_bias_metrics
    multiclass_equality_of_opp
    multiclass_statistical_parity
    multiclass_true_rates
    precision_matrix
    recall_matrix

**Regression**

.. autosummary::
    :nosignatures:
    :template: function.rst
    :toctree: .generated/

    avg_score_diff
    correlation_diff
    disparate_impact_regression
    mae_ratio
    max_statistical_parity
    no_disparate_impact_level
    regression_bias_metrics
    rmse_ratio
    statistical_parity_auc
    statistical_parity_regression
    zscore_diff

**Clustering**

.. autosummary::
    :nosignatures:
    :template: function.rst
    :toctree: .generated/

    cluster_balance
    cluster_dist_entropy
    cluster_dist_kl
    cluster_dist_l1
    clustering_bias_metrics
    min_cluster_ratio
    silhouette_diff
    social_fairness_ratio

**Recommender**

.. autosummary::
    :nosignatures:
    :template: function.rst
    :toctree: .generated/

    aggregate_diversity
    avg_f1_ratio
    avg_precision_ratio
    avg_recall_ratio
    avg_recommendation_popularity
    exposure_entropy
    exposure_kl
    exposure_l1
    gini_index
    mad_score
    recommender_bias_metrics
    recommender_mae_ratio
    recommender_rmse_ratio
