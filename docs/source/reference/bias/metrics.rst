Metrics
=======

`holisticai.bias.metrics` is a python module measuring *bias* in algorithms. Metrics are included for *classification*, *regression*, *clustering*, *recommender* and *multiclass* tasks.

.. _binary_classification:

Binary Classification
----------------------

.. autosummary::
    :toctree: .generated/

    holisticai.bias.metrics.abroca
    holisticai.bias.metrics.accuracy_diff
    holisticai.bias.metrics.average_odds_diff
    holisticai.bias.metrics.classification_bias_metrics
    holisticai.bias.metrics.cohen_d
    holisticai.bias.metrics.disparate_impact
    holisticai.bias.metrics.equal_opportunity_diff
    holisticai.bias.metrics.false_negative_rate_diff
    holisticai.bias.metrics.false_positive_rate_diff
    holisticai.bias.metrics.four_fifths
    holisticai.bias.metrics.statistical_parity
    holisticai.bias.metrics.true_negative_rate_diff
    holisticai.bias.metrics.z_test_diff
    holisticai.bias.metrics.z_test_ratio


.. _multiclass_classification:

Multiclass Classification
-------------------------

.. autosummary::
    :toctree: .generated/

    holisticai.bias.metrics.accuracy_matrix
    holisticai.bias.metrics.confusion_matrix
    holisticai.bias.metrics.confusion_tensor
    holisticai.bias.metrics.frequency_matrix
    holisticai.bias.metrics.multiclass_average_odds
    holisticai.bias.metrics.multiclass_bias_metrics
    holisticai.bias.metrics.multiclass_equality_of_opp
    holisticai.bias.metrics.multiclass_statistical_parity
    holisticai.bias.metrics.multiclass_true_rates
    holisticai.bias.metrics.precision_matrix
    holisticai.bias.metrics.recall_matrix

.. _regression:

Regression
----------

.. autosummary::
    :toctree: .generated/

    holisticai.bias.metrics.avg_score_diff
    holisticai.bias.metrics.correlation_diff
    holisticai.bias.metrics.disparate_impact_regression
    holisticai.bias.metrics.mae_ratio
    holisticai.bias.metrics.max_statistical_parity
    holisticai.bias.metrics.no_disparate_impact_level
    holisticai.bias.metrics.regression_bias_metrics
    holisticai.bias.metrics.rmse_ratio
    holisticai.bias.metrics.statistical_parity_auc
    holisticai.bias.metrics.statistical_parity_regression
    holisticai.bias.metrics.zscore_diff

.. _clustering:

Clustering
----------

.. autosummary::
    :toctree: .generated/

    holisticai.bias.metrics.cluster_balance
    holisticai.bias.metrics.cluster_dist_entropy
    holisticai.bias.metrics.cluster_dist_kl
    holisticai.bias.metrics.cluster_dist_l1
    holisticai.bias.metrics.clustering_bias_metrics
    holisticai.bias.metrics.min_cluster_ratio
    holisticai.bias.metrics.silhouette_diff
    holisticai.bias.metrics.social_fairness_ratio

.. _recommender:

Recommender
-----------

.. autosummary::
    :toctree: .generated/

    holisticai.bias.metrics.aggregate_diversity
    holisticai.bias.metrics.avg_f1_ratio
    holisticai.bias.metrics.avg_precision_ratio
    holisticai.bias.metrics.avg_recall_ratio
    holisticai.bias.metrics.avg_recommendation_popularity
    holisticai.bias.metrics.exposure_entropy
    holisticai.bias.metrics.exposure_kl
    holisticai.bias.metrics.exposure_l1
    holisticai.bias.metrics.gini_index
    holisticai.bias.metrics.mad_score
    holisticai.bias.metrics.recommender_bias_metrics
    holisticai.bias.metrics.recommender_mae_ratio
    holisticai.bias.metrics.recommender_rmse_ratio
