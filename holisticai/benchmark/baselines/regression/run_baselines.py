import os

import pandas as pd

from holisticai.benchmark.tasks import get_task
from holisticai.bias.mitigation import (  
    ExponentiatedGradientReduction,
    GridSearchReduction,
    PluginEstimationAndCalibration,
    WassersteinBarycenter,
    CorrelationRemover,
    DisparateImpactRemover,
)

MITIGATORS_PREPROCESSING = [
    CorrelationRemover(),
    DisparateImpactRemover(repair_level=0.0),
]

MITIGATORS_INPROCESSING = [
    ExponentiatedGradientReduction(constraints="BoundedGroupLoss", loss='Square', min_val=-0.1, max_val=1.3, upper_bound=0.001,),
    GridSearchReduction(constraints="BoundedGroupLoss", loss='Square', min_val=-0.1, max_val=1.3, grid_size=20),
]

MITIGATORS_POSTPROCESSING = [
    PluginEstimationAndCalibration(),
    WassersteinBarycenter(),
]

MITIGATORS = {
    "preprocessing": MITIGATORS_PREPROCESSING,
    "inprocessing": MITIGATORS_INPROCESSING,
    "postprocessing": MITIGATORS_POSTPROCESSING,
}


def __hai_bench__():
    task = get_task("regression")
    for type in ["inprocessing", "preprocessing", "postprocessing"]:
        dataframe = pd.DataFrame()
        for mitigator in MITIGATORS[type]:
            task.run_benchmark(mitigator=mitigator, type=type)
            data = task.evaluate_table(ranking=False, highlight=False, tab=False)
            dataframe = (
                pd.concat([dataframe, data], axis=0)
                .sort_values(by=["Dataset", "RFS"], ascending=False)
                .reset_index(drop=True)
            )
        benchmarkdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataframe.to_parquet(
            f"{benchmarkdir}/regression/{type}_benchmark.parquet"
        )


__hai_bench__()
