In-processing Methods
=====================

In-processing techniques modify the learning algorithm itself to reduce bias during the model training phase. These methods work by incorporating fairness constraints or objectives directly into the training process, ensuring that the model learns to make unbiased decisions. In-processing methods can include adversarial debiasing, fairness constraints, and regularization techniques. By addressing bias during the training phase, these methods aim to create models that are intrinsically fair and less likely to produce biased outcomes.

Here are the in-processing methods implemented in the Holistic AI package:

    .. toctree::
        :maxdepth: 1

        inprocessing/bc_exp_grad_grid_search_exponentiated_gradient_reduction.rst
        inprocessing/bc_exp_grad_grid_search_grid_search.rst
        inprocessing/bc_meta_fair_classifier_rho_fair.rst
        inprocessing/bc_prejudice_remover_prejudice_remover_regularizer.rst
        inprocessing/c_fair_k_center_fair_k_center.rst
        inprocessing/c_fair_k_median_fair_k_median.rst
        inprocessing/c_fairlet_clustering_fairlet_decomposition.rst
        inprocessing/c_variational_fair_clustering_variational_fair_clustering.rst
        inprocessing/mc_fair_scoring_classifier_fairscoringsystems.rst
        inprocessing/rs_blind_spot_aware_blind_spot_aware_matrix_factorization.rst
        inprocessing/rs_popularity_propensity_propensity_scored_recommendations.rst
        inprocessing/rs_two_sided_fairness_fairrec_two_sided_fairness.rst
        inprocessing/bc_adversarial_debiasing_adversarial_debiasing.rst
        inprocessing/rs_popularity_propensity_matrix_factorization.rst