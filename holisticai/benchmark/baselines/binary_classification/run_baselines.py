import os
import sys
sys.path.append('./././')
import pandas as pd

from holisticai.benchmark.tasks import get_task
from holisticai.bias.mitigation import (  # AdversarialDebiasing,
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
    # AdversarialDebiasing(),
    # ExponentiatedGradientReduction(verbose=1),
    GridSearchReduction(constraints="DemographicParity", loss='Square', min_val=-0.1, max_val=1.3, grid_size=20),
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
    for type in ["inprocessing", "preprocessing", "postprocessing"]:
        dataframe = pd.DataFrame()
        for mitigator in MITIGATORS[type]:
            task.run_benchmark(custom_mitigator=mitigator, type=type, _implemented=True)
            data = task.evaluate_table(ranking=False, highlight=False, tab=False)
            dataframe = (
                pd.concat([dataframe, data], axis=0)
                .sort_values(by=["Dataset", "AFS"], ascending=False)
                .reset_index(drop=True)
            )
        benchmarkdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataframe.to_parquet(
            f"{benchmarkdir}/binary_classification/{type}_benchmark.parquet"
        )


__hai_bench__()
