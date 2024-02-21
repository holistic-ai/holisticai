import pandas as pd
import os 

from holisticai.benchmark.tasks import get_task
from holisticai.bias.mitigation import (
    #AdversarialDebiasing,
    CalibratedEqualizedOdds,
    CorrelationRemover,
    DisparateImpactRemover,
    EqualizedOdds,
    ExponentiatedGradientReduction,
    GridSearchReduction,
    LearningFairRepresentation,
    LPDebiaserBinary,
    MetaFairClassifier,
    MLDebiaser,
    PrejudiceRemover,
    RejectOptionClassification,
    Reweighing,
)

MITIGATORS_PREPROCESSING = [
    CorrelationRemover(),
    DisparateImpactRemover(),
    LearningFairRepresentation(verbose=0),
    Reweighing(),
]

MITIGATORS_INPROCESSING = [
    #AdversarialDebiasing(),
    #ExponentiatedGradientReduction(verbose=1),
    GridSearchReduction(),
    MetaFairClassifier(),
    PrejudiceRemover(),
]

MITIGATORS_POSTPROCESSING = [
    CalibratedEqualizedOdds(cost_constraint="fnr"),
    EqualizedOdds(solver="highs", seed=42),
    LPDebiaserBinary(),
    MLDebiaser(),
    RejectOptionClassification(metric_name="Statistical parity difference"),
]

MITIGATORS = {
    "preprocessing": MITIGATORS_PREPROCESSING,
    "inprocessing": MITIGATORS_INPROCESSING,
    "postprocessing": MITIGATORS_POSTPROCESSING,
}


def __hai_bench__():
    task = get_task("binary_classification")
    for type in ["postprocessing"]:#, "preprocessing"]:#["inprocessing", "preprocessing", "postprocessing"]:
        dataframe = pd.DataFrame()
        for mitigator in MITIGATORS[type]:
            task.run_benchmark(mitigator=mitigator, type=type)
            data = task.evaluate_table(ranking=False, highlight=False, tab=False)
            dataframe = (
                pd.concat([dataframe, data], axis=0)
                .sort_values(by=["Dataset", "AFS"], ascending=False)
                .reset_index(drop=True)
            )
        benchmarkdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataframe.to_parquet(
            f"{benchmarkdir}/baselines/{type}_binary_classification_benchmark.parquet"
        )


__hai_bench__()
