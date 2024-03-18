import os
import pandas as pd

from holisticai.benchmark.tasks import get_task
from holisticai.bias.mitigation import (  
    FairKCenterClustering,
    FairKmedianClustering,
    FairletClustering,
    VariationalFairClustering,
    MCMF,
    FairletClusteringPreprocessing
)

MITIGATORS_PREPROCESSING = [
    FairletClusteringPreprocessing()
]

MITIGATORS_INPROCESSING = [
    FairKCenterClustering(req_nr_per_group=(1,1), nr_initially_given=0, seed=1234),
#    FairKmedianClustering(n_clusters=3, strategy='GA'),
    FairletClustering(n_clusters=3),
    VariationalFairClustering(n_clusters=3),
]

MITIGATORS_POSTPROCESSING = [
    MCMF(metric='L1', verbose=0),
]

MITIGATORS = {
    "preprocessing": MITIGATORS_PREPROCESSING,
    "inprocessing": MITIGATORS_INPROCESSING,
    "postprocessing": MITIGATORS_POSTPROCESSING,
}

EXTRA_PRED_PARAMS = {
    "MCMF": {'bm__centroids':"cluster_centers_"},
}


def __hai_bench__():
    task = get_task("clustering")
    for type in ["inprocessing", "preprocessing", "postprocessing"]:
        dataframe = pd.DataFrame()
        for mitigator in MITIGATORS[type]:
            if mitigator.__class__.__name__ in EXTRA_PRED_PARAMS:
                extra_pred_params = EXTRA_PRED_PARAMS[mitigator.__class__.__name__]
            else:
                extra_pred_params = None
            task.run_benchmark(mitigator=mitigator, type=type, extra_pred_params=extra_pred_params)
            data = task.evaluate_table(ranking=False, highlight=False, tab=False)
            dataframe = (
                pd.concat([dataframe, data], axis=0)
                .sort_values(by=["Dataset", "Cluster Balance"], ascending=False)
                .reset_index(drop=True)
            )
        benchmarkdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataframe.to_parquet(
            f"{benchmarkdir}/clustering/{type}_benchmark.parquet"
        )


__hai_bench__()