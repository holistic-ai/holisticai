Bias
====

Measuring
---------

Here are some tutorials where different systems are assessed using the metrics.

- `Measuring Bias Classification <measuring_bias_tutorials/measuring_bias_classification.ipynb>`_
- `Measuring Bias Regression <measuring_bias_tutorials/measuring_bias_regression.ipynb>`_
- `Measuring Bias Recommender <measuring_bias_tutorials/measuring_bias_recommender.ipynb>`_
- `Measuring Bias Clustering <measuring_bias_tutorials/measuring_bias_clustering.ipynb>`_
- `Measuring Bias Multiclass <measuring_bias_tutorials/measuring_bias_multiclass.ipynb>`_

Mitigation
----------

Here are some tutorials where different systems are fitting using the mitigators.

- Binary Classification
    - Preprocessing
        - `Mitigating Bias using Correlation Remover <notebooks/bias/mitigating_bias_tutorials/binary_classification/preprocessing/correlation_remover.ipynb>`_
        - `Mitigating Bias using Disparate Impact <notebooks/bias/mitigating_bias_tutorials/binary_classification/preprocessing/disparate_impact.ipynb>`_
        - `Mitigating Bias using Learning Fair Representation <notebooks/bias/mitigating_bias_tutorials/binary_classification/preprocessing/learning_fair_representation.ipynb>`_
        - `Mitigating Bias using Reweighing <notebooks/bias/mitigating_bias_tutorials/binary_classification/preprocessing/reweighing.ipynb>`_

    - Inprocessing
        - `Mitigating Bias using Adversariala Debiasing <notebooks/biasmitigating_bias_tutorials/binary_classification/inprocessing/adversarial_debiasing.ipynb>`_
        - `Mitigating Bias using Exponientiated Gradient <notebooks/biasmitigating_bias_tutorials/binary_classification/inprocessing/exponientiated_gradient.ipynb>`_
        - `Mitigating Bias using Grid Search Reduction <notebooks/biasmitigating_bias_tutorials/binary_classification/inprocessing/grid_search_reduction.ipynb>`_
        - `Mitigating Bias using Meta Fair Classifier <notebooks/biasmitigating_bias_tutorials/binary_classification/inprocessing/meta_fair_classifier.ipynb>`_
        - `Mitigating Bias using Prejudice Remover <notebooks/biasmitigating_bias_tutorials/binary_classification/inprocessing/prejudice_remover.ipynb>`_

    - Postprocessing
        - `Mitigating Bias using Calibrated Equalize Odds Debiasing <notebooks/biasmitigating_bias_tutorials/binary_classification/postprocessing/calibrated_equalized_odds.ipynb>`_
        - `Mitigating Bias using Equaliized Odds <notebooks/biasmitigating_bias_tutorials/binary_classification/postprocessing/equalized_odds.ipynb>`_
        - `Mitigating Bias using LP Debiaser <notebooks/biasmitigating_bias_tutorials/binary_classification/postprocessing/lp_debiaser.ipynb>`_
        - `Mitigating Bias using ML Debiaser <notebooks/biasmitigating_bias_tutorials/binary_classification/postprocessing/ml_debiaser.ipynb>`_
        - `Mitigating Bias using Reject Option Classification <notebooks/biasmitigating_bias_tutorials/binary_classification/postprocessing/reject_option_classification.ipynb>`_

- Regression
    - Preprocessing
        - `Mitigating Bias using Correlation Remover <notebooks/biasmitigating_bias_tutorials/regression/preprocessing/correlation_remover.ipynb>`_
        - `Mitigating Bias using Disparate Impact <notebooks/biasmitigating_bias_tutorials/regression/preprocessing/disparate_impact_remover.ipynb>`_

    - Inprocessing
        - `Mitigating Bias using Exponientiated Gradient <notebooks/biasmitigating_bias_tutorials/regression/inprocessing/exponientiated_gradient.ipynb>`_
        - `Mitigating Bias using Grid Search Reduction <notebooks/biasmitigating_bias_tutorials/regression/inprocessing/grid_search_reduction.ipynb>`_

    - Postprocessing
        - `Mitigating Bias using Plugin Estimator and Calibrator <notebooks/biasmitigating_bias_tutorials/regression/postprocessing/plugin_estimator_and_calibrator.ipynb>`_
        - `Mitigating Bias using Wasserstein Barycenters <notebooks/biasmitigating_bias_tutorials/regression/postprocessing/wasserstein_barycenters.ipynb>`_

- Multiclass Classification
    - Preprocessing
        - `Mitigating Bias using Correlation Remover <notebooks/biasmitigating_bias_tutorials/multi_classification/preprocessing/correlation_remover.ipynb>`_
        - `Mitigating Bias using Disparate Impact <notebooks/biasmitigating_bias_tutorials/multi_classification/preprocessing/disparate_impact.ipynb>`_
        - `Mitigating Bias using Reweighing <notebooks/biasmitigating_bias_tutorials/multi_classification/preprocessing/reweighing.ipynb>`_

    - Inprocessing
        - `Mitigating Bias using Fair Scoring Classifier <notebooks/biasmitigating_bias_tutorials/multi_classification/inprocessing/fair_scoring_classifier.ipynb>`_

    - Postprocessing
        - `Mitigating Bias using LP Debiaser <notebooks/biasmitigating_bias_tutorials/multi_classification/postprocessing/lp_debiaser.ipynb>`_
        - `Mitigating Bias using ML Debiaser <notebooks/biasmitigating_bias_tutorials/multi_classification/postprocessing/ml_debiaser.ipynb>`_
        
- Recommender Systems
    - Preprocessing
        - `Mitigating Bias using Disparate Impact Remover <mitigating_bias_tutorials/recommender_systems/preprocessing/disparate_impact_remover.ipynb>`_

    - Inprocessing
        - `Mitigating Bias using Blind Spot Aware <notebooks/biasmitigating_bias_tutorials/recommender_systems/inprocessing/blind_spot_aware.ipynb>`_
        - `Mitigating Bias using Debiasing Learning <notebooks/biasmitigating_bias_tutorials/recommender_systems/inprocessing/debiasing_learning.ipynb>`_
        - `Mitigating Bias using Popularity Propensity <notebooks/biasmitigating_bias_tutorials/recommender_systems/inprocessing/popularity_propensity.ipynb>`_
        - `Mitigating Bias using Two Side Fairness <notebooks/biasmitigating_bias_tutorials/recommender_systems/inprocessing/two_sided_fairness.ipynb>`_

    - Postprocessing
        - `Mitigating Bias using Debiasing Exposure <notebooks/biasmitigating_bias_tutorials/recommender_systems/postprocessing/debiasing_exposure.ipynb>`_
        - `Mitigating Bias using Fair Top-K <notebooks/biasmitigating_bias_tutorials/recommender_systems/postprocessing/fair_top_k.ipynb>`_
        
- Clustering
    - Preprocessing
        - `Mitigating Bias using Fairlet Clustering <notebooks/biasmitigating_bias_tutorials/clustering/preprocessing/fairlet_clustering_preprocessing.ipynb>`_

    - Inprocessing
        - `Mitigating Bias using Fair-K Center Clustering <notebooks/biasmitigating_bias_tutorials/clustering/inprocessing/fair_k_center_clustering.ipynb>`_
        - `Mitigating Bias using Fair-K Mediam Clustering <notebooks/biasmitigating_bias_tutorials/clustering/inprocessing/fair_k_median_clustering.ipynb>`_
        - `Mitigating Bias using Fairlet Clustering <notebooks/biasmitigating_bias_tutorials/clustering/inprocessing/fairlet_clustering.ipynb>`_
        - `Mitigating Bias using Variational Fair Clustering <notebooks/biasmitigating_bias_tutorials/clustering/inprocessing/variational_fair_clustering.ipynb>`_

    - Postprocessing
        - `Mitigating Bias using MCMF Clustering <notebooks/biasmitigating_bias_tutorials/clustering/postprocessing/mcmf_clustering.ipynb>`_
