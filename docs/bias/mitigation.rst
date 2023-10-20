Mitigation
==========

`holisticai.bias.mitigation` is a python module mitigating *bias* in algorithms. Our classes cover *pre-processing*, *in-processing* and *post-processing*.

.. _preprocessing:

Pre-processing
--------------

.. autosummary:: 
    :toctree: .generated/

    holisticai.bias.mitigation.Reweighing
    holisticai.bias.mitigation.LearningFairRepresentation
    holisticai.bias.mitigation.CorrelationRemover
    holisticai.bias.mitigation.FairletClusteringPreprocessing
    holisticai.bias.mitigation.DisparateImpactRemover

.. _inprocessing:

In-processing
--------------

.. autosummary:: 
    :toctree: .generated/

    holisticai.bias.mitigation.ExponentiatedGradientReduction
    holisticai.bias.mitigation.GridSearchReduction
    holisticai.bias.mitigation.MetaFairClassifier
    holisticai.bias.mitigation.PrejudiceRemover
    holisticai.bias.mitigation.FairKCenterClustering
    holisticai.bias.mitigation.FairKmedianClustering
    holisticai.bias.mitigation.FairletClustering
    holisticai.bias.mitigation.VariationalFairClustering
    holisticai.bias.mitigation.AdversarialDebiasing
    holisticai.bias.mitigation.FairScoreClassifier
    holisticai.bias.mitigation.BlindSpotAwareMF
    holisticai.bias.mitigation.PopularityPropensityMF
    holisticai.bias.mitigation.FairRec

.. _postprocessing:

Post-processing
---------------

.. autosummary:: 
    :toctree: .generated/
    
    holisticai.bias.mitigation.CalibratedEqualizedOdds
    holisticai.bias.mitigation.EqualizedOdds
    holisticai.bias.mitigation.RejectOptionClassification
    holisticai.bias.mitigation.LPDebiaserBinary
    holisticai.bias.mitigation.LPDebiaserMulticlass
    holisticai.bias.mitigation.MLDebiaser
    holisticai.bias.mitigation.PluginEstimationAndCalibration
    holisticai.bias.mitigation.WassersteinBarycenter
    holisticai.bias.mitigation.DebiasingExposure
    holisticai.bias.mitigation.FairTopK
    holisticai.bias.mitigation.MCMF
    holisticai.bias.mitigation.DisparateImpactRemoverRS
