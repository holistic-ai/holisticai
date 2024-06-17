Bias Mitigation
==========

`holisticai.mitigation.bias` is a python module mitigating *bias* in algorithms. Our classes cover *pre-processing*, *in-processing* and *post-processing*.

.. _preprocessing:

Pre-processing
--------------

.. autosummary:: 
    :toctree: .generated/

    holisticai.mitigation.bias.Reweighing
    holisticai.mitigation.bias.LearningFairRepresentation
    holisticai.mitigation.bias.CorrelationRemover
    holisticai.mitigation.bias.FairletClusteringPreprocessing
    holisticai.mitigation.bias.DisparateImpactRemover

.. _inprocessing:

In-processing
--------------

.. autosummary:: 
    :toctree: .generated/

    holisticai.mitigation.bias.ExponentiatedGradientReduction
    holisticai.mitigation.bias.GridSearchReduction
    holisticai.mitigation.bias.MetaFairClassifier
    holisticai.mitigation.bias.PrejudiceRemover
    holisticai.mitigation.bias.FairKCenterClustering
    holisticai.mitigation.bias.FairKMedianClustering
    holisticai.mitigation.bias.FairletClustering
    holisticai.mitigation.bias.VariationalFairClustering
    holisticai.mitigation.bias.FairScoreClassifier
    holisticai.mitigation.bias.BlindSpotAwareMF
    holisticai.mitigation.bias.PopularityPropensityMF
    holisticai.mitigation.bias.FairRec

.. _postprocessing:

Post-processing
---------------

.. autosummary:: 
    :toctree: .generated/
    
    holisticai.mitigation.bias.CalibratedEqualizedOdds
    holisticai.mitigation.bias.EqualizedOdds
    holisticai.mitigation.bias.RejectOptionClassification
    holisticai.mitigation.bias.LPDebiaserBinary
    holisticai.mitigation.bias.LPDebiaserMulticlass
    holisticai.mitigation.bias.MLDebiaser
    holisticai.mitigation.bias.PluginEstimationAndCalibration
    holisticai.mitigation.bias.WassersteinBarycenter
    holisticai.mitigation.bias.DebiasingExposure
    holisticai.mitigation.bias.FairTopK
    holisticai.mitigation.bias.MCMF
    holisticai.mitigation.bias.DisparateImpactRemoverRS
