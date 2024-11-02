from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression

from holisticai.bias.metrics import (
    balanced_fairness_score,
    balanced_fairness_score_error,
    balanced_fairness_score_multiclass,
    social_fairness_ratio,
)
from holisticai.bias.mitigation import (
    MCMF,
    AdversarialDebiasing,
    CalibratedEqualizedOdds,
    CorrelationRemover,
    DisparateImpactRemover,
    EqualizedOdds,
    ExponentiatedGradientReduction,
    FairKCenterClustering,
    FairletClusteringPreprocessing,
    FairScoreClassifier,
    GridSearchReduction,
    LearningFairRepresentation,
    LPDebiaserBinary,
    LPDebiaserMulticlass,
    MLDebiaser,
    PluginEstimationAndCalibration,
    PrejudiceRemover,
    RejectOptionClassification,
    Reweighing,
    WassersteinBarycenter,
)

DATASETS = {
    "binary_classification": [
        "compass_sex",
        "compass_race",
        "adult_sex",
        "adult_race",
        "german_credit_sex",
        "clinical_records_sex",
        "bank_marketing_marital",
        "law_school_sex",
        "law_school_race",
        "diabetes_sex",
        "diabetes_race",
        "census_kdd_sex",
        "acsincome_sex",
        "acsincome_race",
        "acspublic_sex",
        "acspublic_race",
    ],
    "multiclass": [
        "us_crime_multiclass_race",
        "student_multiclass_sex",
        "student_multiclass_address",
    ],
    "regression": [
        "us_crime_race",
        "student_sex",
        "student_address",
    ],
    "clustering": [
        "clinical_records_sex",
        "student_sex",
        "student_address",
        "german_credit_sex",
        "compass_sex",
        "compass_race",
        "adult_sex",
        "adult_race",
    ],
}

METRICS = {
    "binary_classification": balanced_fairness_score,
    "multiclass": balanced_fairness_score_multiclass,
    "regression": balanced_fairness_score_error,
    "clustering": social_fairness_ratio,
}

MODELS = {
    "binary_classification": LogisticRegression(random_state=42),
    "multiclass": LogisticRegression(multi_class="multinomial", solver="newton-cg"),
    "regression": LinearRegression(),
    "clustering": KMeans(n_clusters=3, random_state=42),
}


MITIGATORS = {
    "binary_classification": {
        "preprocessing": [CorrelationRemover(), DisparateImpactRemover(), LearningFairRepresentation(), Reweighing()],
        "inprocessing": [
            AdversarialDebiasing,
            GridSearchReduction,
            PrejudiceRemover,
        ],
        "postprocessing": [
            CalibratedEqualizedOdds(cost_constraint="fnr"),
            EqualizedOdds(solver="highs", seed=42),
            RejectOptionClassification(metric_name="Statistical parity difference"),
            LPDebiaserBinary(),
            MLDebiaser(),
        ],
    },
    "multiclass": {
        "preprocessing": [CorrelationRemover(), DisparateImpactRemover(), Reweighing()],
        "inprocessing": [FairScoreClassifier],
        "postprocessing": [
            LPDebiaserMulticlass(),
            MLDebiaser(),
        ],
    },
    "regression": {
        "preprocessing": [CorrelationRemover(), DisparateImpactRemover()],
        "inprocessing": [
            ExponentiatedGradientReduction,
            GridSearchReduction,
        ],
        "postprocessing": [
            PluginEstimationAndCalibration(),
            WassersteinBarycenter(),
        ],
    },
    "clustering": {
        "preprocessing": [
            FairletClusteringPreprocessing(seed=42),
        ],
        "inprocessing": [
            FairKCenterClustering,
        ],
        "postprocessing": [
            MCMF(metric="L1", solver="highs-ipm"),
        ],
    },
}
