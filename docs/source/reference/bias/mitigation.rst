:py:mod:`holisticai.bias.mitigation`
====================================

.. automodule:: holisticai.bias.mitigation
    :no-members:
    :no-inherited-members:

**Pre-processing**

.. autosummary:: 
    :nosignatures:
    :template: class.rst
    :toctree: .generated/

    Reweighing
    LearningFairRepresentation
    CorrelationRemover
    FairletClusteringPreprocessing
    DisparateImpactRemover

**In-processing**

.. autosummary:: 
    :toctree: .generated/
    :nosignatures:
    :template: class.rst

    ExponentiatedGradientReduction
    GridSearchReduction
    MetaFairClassifier
    PrejudiceRemover
    FairKCenterClustering
    FairKMedianClustering
    FairletClustering
    VariationalFairClustering
    FairScoreClassifier
    BlindSpotAwareMF
    PopularityPropensityMF
    FairRec

**Post-processing**

.. autosummary:: 
    :toctree: .generated/
    :nosignatures:
    :template: class.rst
    
    CalibratedEqualizedOdds
    EqualizedOdds
    RejectOptionClassification
    LPDebiaserBinary
    LPDebiaserMulticlass
    MLDebiaser
    PluginEstimationAndCalibration
    WassersteinBarycenter
    DebiasingExposure
    FairTopK
    MCMF
    DisparateImpactRemoverRS
