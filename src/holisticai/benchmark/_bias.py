from __future__ import annotations

import os
import time
from typing import Callable

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from holisticai.benchmark._config import DATASETS, METRICS, MITIGATORS
from holisticai.datasets._load_dataset import _load_dataset_benchmark
from holisticai.pipeline import Pipeline

MODELS = {
    "binary_classification": LogisticRegression(random_state=42),
    "multiclass_classification": LogisticRegression(multi_class='multinomial', solver='newton-cg'),
    "regression": LinearRegression(),
}

class BiasMitigationBenchmark:

    METRICS = METRICS
    DATASETS = DATASETS
    MITIGATORS = MITIGATORS
    MODELS = MODELS

    def __init__(self, task_type: str, stage: str):
        if task_type not in self.DATASETS:
            raise ValueError(f"Invalid task type. Choose from {list(self.DATASETS.keys())}")
        self.task_type = task_type
        self.metric = self.METRICS[task_type]
        self.stage = stage

    def get_datasets(self) -> list[str]:
        return self.DATASETS[self.task_type]

    def get_mitigators(self) -> dict[str, list[Callable]]:
        if self.stage:
            if self.stage not in self.MITIGATORS[self.task_type]:
                raise ValueError(f"Invalid stage. Choose from {list(self.MITIGATORS[self.task_type].keys())}")
            return {self.stage: self.MITIGATORS[self.task_type][self.stage]}
        return self.MITIGATORS[self.task_type]

    def get_table(self):
        osdir = os.path.dirname(os.path.abspath(__file__))
        data = pd.read_csv(os.path.join(osdir, "results", self.task_type, self.stage, "benchmark.csv"))
        return data

    def get_plot(self):
        pass

    def run(self, custom_mitigator):
        return # results_table (position in ranking)

    def submit(self):
        "forms with the repository/notebook"

    def _build_benchmark(self, datasets, mitigators, **kwargs):

        results = pd.DataFrame({"dataset": [], "mitigator": [], "metric": [], "time":[], "value": []})
        for dataset_name in datasets:
            data = _load_dataset_benchmark(dataset_name=dataset_name)
            data_split = data.train_test_split(test_size=0.2, random_state=42)
            train = data_split['train']
            test = data_split['test']
            for mitigator in mitigators[self.stage]:
                mitigator_name = mitigator.__class__.__name__
                start_time = time.time()
                if self.stage == "preprocessing":
                    pipeline = Pipeline(steps=[("scalar", StandardScaler()),
                                               ("bm_preprocessing", mitigator),
                                               ("estimator", self.MODELS[self.task_type]),])

                elif self.stage == "postprocessing":
                    pipeline = Pipeline(steps=[("scalar", StandardScaler()),
                                               ("estimator", self.MODELS[self.task_type]),
                                               ("bm_posprocessing", mitigator)])

                elif self.stage == "inprocessing":
                    mitigator_name = mitigator.__name__
                    if mitigator_name == "AdversarialDebiasing":
                        in_mitigator = mitigator(features_dim=train['X'].shape[1], batch_size=512, hidden_size=64,
                                                 adversary_loss_weight=3, verbose=1,
                                                 use_debias=True, seed=42).transform_estimator()
                    elif mitigator_name == "GridSearchReduction" and self.task_type == "binary_classification":
                        in_mitigator = mitigator(constraints="EqualizedOdds",
                                                 grid_size=20).transform_estimator(self.MODELS[self.task_type])
                    elif mitigator_name == "GridSearchReduction" and self.task_type == "regression":
                        in_mitigator = mitigator(constraints="BoundedGroupLoss",
                                                 loss='Square', min_val=-0.1, max_val=0.1,
                                                 grid_size=50).transform_estimator(self.MODELS[self.task_type])
                    elif mitigator_name == "PrejudiceRemover":
                        in_mitigator = mitigator(maxiter=100, fit_intercept=True,
                                                 verbose=1, print_interval=1).transform_estimator(self.MODELS[self.task_type])
                    elif mitigator_name == "ExponentiatedGradientReduction":
                        in_mitigator = mitigator(constraints="BoundedGroupLoss",
                                                 loss='Square', min_val=-0.1, max_val=1.3,
                                                 upper_bound=0.001).transform_estimator(self.MODELS[self.task_type])
                    else:
                        in_mitigator = mitigator(**kwargs)
                    pipeline = Pipeline(steps=[("scalar", StandardScaler()),
                                               ("bm_inprocessing", in_mitigator)])

                pipeline.fit(train['X'], train['y'], bm__group_a=train['group_a'], bm__group_b=train['group_b'])
                y_pred_pipeline = pipeline.predict(test['X'], bm__group_a=test['group_a'], bm__group_b=test['group_b'])
                end_time = time.time()
                time_pipeline = end_time - start_time

                metric = METRICS[self.task_type](test['group_a'], test['group_b'], y_pred_pipeline)
                results_pipeline = pd.DataFrame({"dataset": [dataset_name],
                                                 "mitigator": [mitigator_name],
                                                 "time": [time_pipeline],
                                                 "value": [metric]})
                results = pd.concat([results, results_pipeline])

        df_time = results.pivot(index='mitigator', columns=['dataset'], values=['time'])
        df_result = results.pivot(index='mitigator', columns=['dataset'], values=['value'])
        df_result.columns = [f'{col[1].upper()}' for col in df_result.columns]
        df_result.insert(0, 'Mean Value', df_result.mean(axis=1))
        df_result.insert(1, 'Std', df_result.std(axis=1))
        df_result.insert(2, 'Mean Time', df_time.mean(axis=1))
        df_result = df_result.sort_values(by='Mean Value', ascending=False)
        return df_result.T
