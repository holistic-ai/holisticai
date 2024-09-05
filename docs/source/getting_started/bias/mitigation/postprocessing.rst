Post-processing Methods
=======================

Post-processing techniques adjust the predictions of the trained model to mitigate bias. These methods involve modifying the outputs or decision thresholds of the model to ensure fair treatment across different groups. Post-processing methods can include calibrated equalized odds, reject option classification, and output adjustment. By addressing bias at the prediction stage, post-processing methods provide a way to correct for any biases that may have been introduced during the training process or that exist in the data. These techniques ensure that the final decisions made by the AI system are fair and equitable for all users.

Here are the in-processing methods implemented in the Holistic AI package:

    .. toctree::
        :maxdepth: 1

        postprocessing/bc_calibrated_eq_odds_postprocessing_calibrated_equalized_odds.rst
        postprocessing/bc_eq_odds_post_processing_equality_of_opportunity.rst
        postprocessing/bc_lp_debiaser_linear_program.rst
        postprocessing/bc_ml_debiaser_rto.rst
        postprocessing/bc_reject_option_classification_reject_option_based_classiﬁcation.rst
        postprocessing/c_mcmf_clustering_mcmf_problem.rst
        postprocessing/mc_lp_debiaser_linear_program.rst
        postprocessing/r_plugin_estimator_and_calibrator_plug_in_estimator_and_recalibration.rst
        .. postprocessing/rs_debiasing_exposure_deltr.rst
        postprocessing/rs_disparate_impact_remover_disparate_impact_remover.rst
        postprocessing/rs_fair_topk_fa*ir_algorithm.rst
        postprocessing/r_wasserstein_barycenters_wasserstein_barycenter.rst