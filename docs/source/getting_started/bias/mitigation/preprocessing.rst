Pre-processing Methods
======================

Pre-processing techniques aim to mitigate bias by transforming the data before it is used to train a model. These methods involve modifying the training data to ensure fair representation and treatment of all groups. By addressing bias at the data level, pre-processing methods can help prevent the introduction of bias into the model and improve the fairness of its predictions. Common pre-processing strategies include reweighting, resampling, data augmentation, and fair representation learning, which work by either balancing the representation of different groups in the dataset or by removing sensitive information that could lead to biased outcomes.

Here are the in-processing methods implemented in the Holistic AI package:

    .. toctree::
        :maxdepth: 1

        preprocessing/c_fairlet_clustering_preprocessing_fairlet_decomposition.rst
        preprocessing/bc_correlation_remover_correlationremover.rst
        preprocessing/bc_disparate_impact_remover_disparate_impact_remover.rst
        preprocessing/bc_learning_fair_representations_lfr.rst
        preprocessing/bc_reweighing_reweighing.rst