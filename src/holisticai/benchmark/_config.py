from holisticai.bias.metrics import statistical_parity, statistical_parity_regression
from holisticai.bias.mitigation import (
    AdversarialDebiasing,
    CalibratedEqualizedOdds,
    CorrelationRemover,
    DisparateImpactRemover,
    EqualizedOdds,
    ExponentiatedGradientReduction,
    GridSearchReduction,
    LPDebiaserBinary,
    MLDebiaser,
    PluginEstimationAndCalibration,
    PrejudiceRemover,
    RejectOptionClassification,
    Reweighing,
    WassersteinBarycenter,
)

DATASETS = {
        "binary_classification": [
            'compass', 'adult', 'german_credit', 'clinical_records', 'bank_marketing', 'law_school',
            'diabetes', 'census_kdd', 'acsincome', 'acspublic',
            ],
        "regression": [
            'us_crime',
            'student',
            ],
}

METRICS = {
    "binary_classification": statistical_parity,
    "regression": statistical_parity_regression,
}

MITIGATORS = {
        "binary_classification": {
                "preprocessing": [
                    CorrelationRemover(),
                    DisparateImpactRemover(),
                    Reweighing()
                    ],
                "inprocessing": [
                    AdversarialDebiasing,
                    GridSearchReduction,
                    PrejudiceRemover,
                    ],
                "postprocessing": [
                    CalibratedEqualizedOdds(cost_constraint="fnr"),
                    EqualizedOdds(solver='highs', seed=42),
                    RejectOptionClassification(metric_name="Statistical parity difference"),
                    LPDebiaserBinary(),
                    MLDebiaser(),
                    ],
                },
        "regression": {
                "preprocessing": [
                    CorrelationRemover(),
                    DisparateImpactRemover()
                    ],
                "inprocessing": [
                    ExponentiatedGradientReduction,
                    GridSearchReduction,
                    ],
                "postprocessing": [
                    PluginEstimationAndCalibration(),
                    WassersteinBarycenter(),
                    ],
                },
        }
