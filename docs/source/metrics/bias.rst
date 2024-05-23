Metrics
=======

`holisticai.metrics.bias` is a python module measuring *bias* in algorithms. Metrics are included for *classification*, *regression*, *clustering*, *recommender* and *multiclass* tasks.

.. _binary classification:

Binary Classification
----------------------

.. autosummary::
    :toctree: .generated/

    holisticai.metrics.bias.abroca
    holisticai.metrics.bias.accuracy_diff
    holisticai.metrics.bias.average_odds_diff
    holisticai.metrics.bias.classification_bias_metrics
    holisticai.metrics.bias.cohen_d
    holisticai.metrics.bias.disparate_impact
    holisticai.metrics.bias.equal_opportunity_diff
    holisticai.metrics.bias.false_negative_rate_diff
    holisticai.metrics.bias.false_positive_rate_diff
    holisticai.metrics.bias.four_fifths
    holisticai.metrics.bias.statistical_parity
    holisticai.metrics.bias.true_negative_rate_diff
    holisticai.metrics.bias.z_test_diff
    holisticai.metrics.bias.z_test_ratio


.. _multiclass classification:

Multiclass Classification
-------------------------

.. autosummary::
    :toctree: .generated/

    holisticai.metrics.bias.accuracy_matrix
    holisticai.metrics.bias.confusion_matrix
    holisticai.metrics.bias.confusion_tensor
    holisticai.metrics.bias.frequency_matrix
    holisticai.metrics.bias.multiclass_average_odds
    holisticai.metrics.bias.multiclass_bias_metrics
    holisticai.metrics.bias.multiclass_equality_of_opp
    holisticai.metrics.bias.multiclass_statistical_parity
    holisticai.metrics.bias.multiclass_true_rates
    holisticai.metrics.bias.precision_matrix
    holisticai.metrics.bias.recall_matrix

.. _regression:

Regression
----------

.. autosummary::
    :toctree: .generated/

    holisticai.metrics.bias.avg_score_diff
    holisticai.metrics.bias.correlation_diff
    holisticai.metrics.bias.disparate_impact_regression
    holisticai.metrics.bias.mae_ratio
    holisticai.metrics.bias.max_statistical_parity
    holisticai.metrics.bias.no_disparate_impact_level
    holisticai.metrics.bias.regression_bias_metrics
    holisticai.metrics.bias.rmse_ratio
    holisticai.metrics.bias.statistical_parity_auc
    holisticai.metrics.bias.statistical_parity_regression
    holisticai.metrics.bias.zscore_diff

.. _clustering:

Clustering
----------

.. autosummary::
    :toctree: .generated/

    holisticai.metrics.bias.cluster_balance
    holisticai.metrics.bias.cluster_dist_entropy
    holisticai.metrics.bias.cluster_dist_kl
    holisticai.metrics.bias.cluster_dist_l1
    holisticai.metrics.bias.clustering_bias_metrics
    holisticai.metrics.bias.min_cluster_ratio
    holisticai.metrics.bias.silhouette_diff
    holisticai.metrics.bias.social_fairness_ratio

.. _recommender:

Recommender
-----------

.. autosummary::
    :toctree: .generated/

    holisticai.metrics.bias.aggregate_diversity
    holisticai.metrics.bias.avg_f1_ratio
    holisticai.metrics.bias.avg_precision_ratio
    holisticai.metrics.bias.avg_recall_ratio
    holisticai.metrics.bias.avg_recommendation_popularity
    holisticai.metrics.bias.exposure_entropy
    holisticai.metrics.bias.exposure_kl
    holisticai.metrics.bias.exposure_l1
    holisticai.metrics.bias.gini_index
    holisticai.metrics.bias.mad_score
    holisticai.metrics.bias.recommender_bias_metrics
    holisticai.metrics.bias.recommender_mae_ratio
    holisticai.metrics.bias.recommender_rmse_ratio
