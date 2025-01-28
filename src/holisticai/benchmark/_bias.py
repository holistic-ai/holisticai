from __future__ import annotations

import logging
import math
import time
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from holisticai.datasets._load_dataset import _load_dataset_benchmark
from holisticai.pipeline import Pipeline
from holisticai.utils._plotting import get_colors
from holisticai.utils.benchmark_config import DATASETS, METRICS, MITIGATORS, MODELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiasMitigationBenchmark:
    METRICS = METRICS
    DATASETS = DATASETS
    MITIGATORS = MITIGATORS
    MODELS = MODELS

    def __init__(self, task_type: str, stage: str):
        if task_type not in self.MODELS:
            raise ValueError(f"Invalid task type. Choose from {list(self.DATASETS.keys())}")
        self.task_type = task_type
        self.metric = self.METRICS[task_type]
        self.stage = stage

    def available_settings(self):
        """
        Get the available settings for the benchmark.

        Returns
        -------
        dict
            Available settings.
        """

        data = list(self.DATASETS[self.task_type])
        mitigator = list(obj.__class__.__name__ for obj in self.MITIGATORS[self.task_type][self.stage])  # noqa: C400

        print(f"Available datasets for {self.task_type}: {data}")  # noqa: T201
        print(f"Available mitigators for {self.task_type} with {self.stage}: {mitigator}")  # noqa: T201

    def get_datasets(self):
        """
        Get the datasets for the given task type.

        Returns
        -------
        dict
            Datasets.
        """
        return self.DATASETS[self.task_type]

    def get_mitigators(self):
        """
        Get the mitigators for the given task type and stage.

        Returns
        -------
        dict
            Mitigators.
        """
        if self.stage:
            if self.stage not in self.MITIGATORS[self.task_type]:
                raise ValueError(f"Invalid stage. Choose from {list(self.MITIGATORS[self.task_type].keys())}")
            return {self.stage: self.MITIGATORS[self.task_type][self.stage]}
        return self.MITIGATORS[self.task_type]

    def get_table(self):
        """
        Get the benchmark results as a table.

        Returns
        -------
        pd.DataFrame
            Benchmark results.
        """
        data = pd.read_parquet(
            f"https://huggingface.co/datasets/holistic-ai/bias_mitigation_benchmark/resolve/main/data/benchmark_{self.task_type}_{self.stage}.parquet",
        )
        return data.set_index("Unnamed: 0").rename_axis("")

    def get_heatmap(self, fig_size=(10, 5), output_path=None):
        """
        Create a heatmap based on the benchmark results.

        Returns
        -------
        fig
            Heatmap.
        """
        plt.style.use("ggplot")
        plt.rc("font", size=14)

        fig, ax = plt.subplots(figsize=fig_size)
        data = self.get_table()[3:].T
        sns.heatmap(data, annot=True, fmt=".4f", cmap=get_colors(len(data.columns)), linewidths=1, ax=ax)
        # color bar title
        cbar = ax.collections[0].colorbar
        cbar.set_label("Balanced Fairness Score (mean)", fontsize=14)
        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        return fig

    def get_plot(self, output_path=None):
        """
        Create a bar plot based on the benchmark results.

        Returns
        -------
        fig
            Plot.
        """
        results = self.get_table()
        plt.style.use("ggplot")
        plt.rc("font", size=14)

        results = results[3:]
        n_datasets = len(results.index)
        n_cols = min(4, n_datasets)  # Maximum 4 columns
        n_rows = math.ceil(n_datasets / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=False)

        axes = axes.flatten() if n_datasets > 1 else [axes]

        for i, (dataset, row) in enumerate(results.iterrows()):
            ax = axes[i]

            bars = ax.barh(row.index, row.values, alpha=0.8, color=get_colors(len(row.values)), label=row.index)

            ax.set_title(dataset, fontsize=12)
            ax.set_xlabel("metric", fontsize=12)
            ax.bar_label(bars, fmt="%.4f", padding=2, label_type="center", fontsize=14)

            if i % n_cols != 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("Model", fontsize=14)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        fig.legend(results.columns, title="Mitigator", loc="center left", bbox_to_anchor=(1, 0.9), ncols=1, fontsize=14)
        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        return fig

    def get_radar(self, figsize=(10, 5), output_path=None):
        """
        Create a radar plot.

        Parameters
        ----------
        figsize : tuple, optional
            Size of the figure. The default is (10, 5).

        Returns
        -------
        ax
            Radar plot.
        """
        tab = self.get_table()
        tab_trans = tab[3:].T.reset_index(names="Mitigators")

        categories = list(tab_trans.columns)[1:]
        N = len(categories)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], categories)
        ax.set_rlabel_position(0)

        min_value = tab[3:].values.min()
        max_value = tab[3:].values.max()
        plt.ylim(min_value, max_value)

        # Add axis lines
        ax.set_rlabel_position(0)
        plt.yticks(
            np.linspace(min_value, max_value, 5),
            [f"{x:.4f}" for x in np.linspace(min_value, max_value, 5)],
            color="grey",
            size=7,
        )
        plt.ylim(min_value, max_value)

        # Add subtle grid lines
        ax.grid(True, linestyle="--", alpha=0.7)

        n_var = len(tab_trans)
        colors = get_colors(n_var)
        for i, color in enumerate(colors):
            values = tab_trans.loc[i].drop("Mitigators").values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle="solid", label=tab_trans.loc[i]["Mitigators"], color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        plt.legend(loc="center left", bbox_to_anchor=(1.3, 0.5))
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
        return ax

    def run(self, custom_mitigator: object = None, custom_dataset=None):
        """
        Run the benchmark for the given task type and stage.

        Parameters
        ----------
        custom_mitigator : object, optional
            Custom mitigator to run the benchmark with. The default is None.
        custom_dataset : object, optional
            Custom dataset to run the benchmark with. The default is None.

        Returns
        -------
        pd.DataFrame
            Results of the benchmark.
        """
        dataset = custom_dataset if custom_dataset is not None else self.get_datasets()
        mitigator = custom_mitigator if custom_mitigator is not None else self.get_mitigators()

        is_custom_dataset = custom_dataset is not None
        is_custom_mitigator = custom_mitigator is not None

        result = self._build_benchmark(
            dataset, mitigator, custom_dataset=is_custom_dataset, custom_mitigator=is_custom_mitigator
        )

        if is_custom_mitigator:
            bench = (
                self.get_table()
                if not is_custom_dataset
                else self._build_benchmark(dataset, self.get_mitigators(), custom_dataset=True, custom_mitigator=False)
            )
            result = pd.concat([bench, result], axis=1)
            result = result.sort_values(by="Mean Score", axis=1, ascending=False)

        return result

    def submit(self):
        import webbrowser

        return webbrowser.open("https://forms.office.com/e/8nLjA7Y38w")

    def _retry_with_backoff(self, func, max_attempts=5, initial_wait=1, backoff_factor=2):
        def wrapper(*args, **kwargs):
            attempts = 0
            wait_time = initial_wait
            while attempts < max_attempts:  # noqa: RET503
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise
                    print(f"Attempt {attempts} failed: {e}. Retrying in {wait_time} seconds...")  # noqa: T201
                    time.sleep(wait_time)
                    wait_time *= backoff_factor

        return wrapper

    def _build_benchmark(self, datasets, mitigators, custom_mitigator=None, custom_dataset=None, **kwargs):
        """
        Run the benchmark for the given task type and stage.

        Parameters
        ----------
        datasets : object
            Datasets to run the benchmark with.
        mitigators : object
            Mitigators to run the benchmark with.
        custom_mitigator : object, optional
            Custom mitigator to run the benchmark with. The default is None.
        custom_dataset : object, optional
            Custom dataset to run the benchmark with. The default is None.

        Returns
        -------
        pd.DataFrame
            Results of the benchmark.
        """
        results = pd.DataFrame({"dataset": [], "mitigator": [], "metric": [], "time": [], "value": []})

        if not custom_dataset:
            for dataset_name in datasets:
                load_dataset = partial(_load_dataset_benchmark, dataset_name)
                load_with_retry = self._retry_with_backoff(load_dataset)
                try:
                    data = load_with_retry()
                except Exception as e:  # noqa: BLE001
                    print(f"Failed to load dataset {dataset_name} after 10 attempts: {e}")  # noqa: T201
                    continue  # Skip to the next dataset if all attempts fail
                data_split = data.train_test_split(test_size=0.2, random_state=42)
                train = data_split["train"]
                test = data_split["test"]

                if custom_mitigator:
                    result_mitigator = self.run_mitigator(mitigators, train, test, dataset_name, **kwargs)
                    results = pd.concat([results, result_mitigator])
                else:
                    for mitigator in mitigators[self.stage]:
                        result_mitigator = self.run_mitigator(mitigator, train, test, dataset_name, **kwargs)
                        results = pd.concat([results, result_mitigator])
        else:
            data = datasets
            dataset_name = "Custom Dataset"
            data_split = data.train_test_split(test_size=0.2, random_state=42)
            train = data_split["train"]
            test = data_split["test"]
            if custom_mitigator:
                result_mitigator = self.run_mitigator(mitigators, train, test, dataset_name, **kwargs)
                results = pd.concat([results, result_mitigator])
            else:
                for mitigator in mitigators[self.stage]:
                    result_mitigator = self.run_mitigator(mitigator, train, test, dataset_name, **kwargs)
                    results = pd.concat([results, result_mitigator])

        df_time = results.pivot(index="mitigator", columns=["dataset"], values=["time"])
        df_result = results.pivot(index="mitigator", columns=["dataset"], values=["value"])
        df_result.columns = [f"{col[1]}" for col in df_result.columns]
        df_result.insert(0, "Mean Score", df_result.mean(axis=1))
        df_result.insert(1, "Std Score", df_result.std(axis=1))
        df_result.insert(2, "Mean Time", df_time.mean(axis=1))
        df_result = df_result.sort_values(by="Mean Score", ascending=False)
        return df_result.T

    def _create_pipeline(self, mitigator: Any, **kwargs) -> Pipeline:
        steps = [("scalar", StandardScaler())]

        if self.stage == "preprocessing":
            steps.extend([("bm_preprocessing", mitigator), ("estimator", self.MODELS[self.task_type])])
        elif self.stage == "postprocessing":
            steps.extend([("estimator", self.MODELS[self.task_type]), ("bm_postprocessing", mitigator)])
        elif self.stage == "inprocessing":
            in_mitigator = self._configure_inprocessing_mitigator(mitigator, **kwargs)
            steps.append(("bm_inprocessing", in_mitigator))

        return Pipeline(steps=steps)

    def _configure_inprocessing_mitigator(self, mitigator: Any, **kwargs) -> Any:
        mitigator_name = mitigator.__name__
        config = {
            "AdversarialDebiasing": lambda: mitigator(
                features_dim=kwargs["train_shape"][1],
                batch_size=512,
                hidden_size=64,
                adversary_loss_weight=3,
                verbose=1,
                use_debias=True,
                seed=42,
            ).transform_estimator(),
            "GridSearchReduction": lambda: mitigator(**self._configure_grid_search_reduction()).transform_estimator(
                self.MODELS[self.task_type]
            ),
            "PrejudiceRemover": lambda: mitigator(
                maxiter=100, fit_intercept=True, verbose=1, print_interval=1
            ).transform_estimator(self.MODELS[self.task_type]),
            "ExponentiatedGradientReduction": lambda: mitigator(
                constraints="BoundedGroupLoss", loss="Square", min_val=-0.1, max_val=1.3, eps=0.01
            ).transform_estimator(self.MODELS[self.task_type]),
            "FairKCenterClustering": lambda: mitigator(req_nr_per_group=(1, 1), nr_initially_given=0, seed=42),
        }
        try:
            return config.get(mitigator_name, lambda: mitigator(**kwargs))()
        except TypeError as e:
            logger.exception(f"Error configuring {mitigator_name}: {e!s}")  # noqa: TRY401
            raise

    def _configure_grid_search_reduction(self) -> Any:
        if self.task_type == "binary_classification":
            args = {"constraints": "EqualizedOdds", "grid_size": 20}
            return args
        if self.task_type == "regression":
            args = {
                "constraints": "BoundedGroupLoss",
                "loss": "Square",
                "min_val": -0.1,
                "max_val": 0.1,
                "grid_size": 50,
            }
            return args
        return None

    def _predict_and_evaluate(self, pipeline: Pipeline, train: dict[str, Any], test: dict[str, Any]) -> tuple[Any, Any]:
        if self.task_type == "clustering":
            return self._handle_clustering(pipeline, train)
        if self.task_type == "multiclass":
            y_pred = pipeline.predict(test["X"], bm__group_a=test["group_a"], bm__group_b=test["group_b"])
            metric = self.METRICS[self.task_type](test["group_a"], y_pred, test["y"])
        else:
            y_pred = pipeline.predict(test["X"], bm__group_a=test["group_a"], bm__group_b=test["group_b"])
            metric = self.METRICS[self.task_type](test["group_a"], test["group_b"], y_pred, test["y"])
        return y_pred, metric

    def _handle_clustering(self, pipeline: Pipeline, train: dict[str, Any]) -> tuple[Any, Any]:
        if self.stage == "postprocessing":
            y_pred = pipeline.predict(
                train["X"], bm__group_a=train["group_a"], bm__group_b=train["group_b"], bm__centroids="cluster_centers_"
            )
            centroids = pipeline["estimator"].cluster_centers_
        else:
            y_pred = pipeline.predict(train["X"], bm__group_a=train["group_a"], bm__group_b=train["group_b"])
            centroids = self._get_centroids(pipeline, train)

        metric = self.METRICS[self.task_type](train["group_a"], train["group_b"], train["X"], centroids)
        return y_pred, metric

    def _get_centroids(self, pipeline: Pipeline, train: dict[str, Any]) -> Any:
        if self.stage == "inprocessing":
            return pipeline["bm_inprocessing"].all_centroids
        if self.stage == "preprocessing":
            model = self.MODELS[self.task_type]
            model.fit(train["X"])
            return model.cluster_centers_
        return None

    def run_mitigator(
        self, mitigator: Any, train: dict[str, Any], test: dict[str, Any], dataset_name: str, **kwargs
    ) -> pd.DataFrame:
        """
        Run the benchmark for the given mitigator.

        Parameters
        ----------
        mitigator : Any
            Mitigator to run the benchmark with.
        train : dict[str, Any]
            Training data.
        test : dict[str, Any]
            Testing data.
        dataset_name : str
            Name of the dataset.

        Returns
        -------
        pd.DataFrame
            Results of the benchmark.
        """
        mitigator_name = mitigator.__class__.__name__ if self.stage != "inprocessing" else mitigator.__name__
        start_time = time.time()

        try:
            pipeline = self._create_pipeline(mitigator, train_shape=train["X"].shape, **kwargs)
            pipeline.fit(train["X"], train["y"], bm__group_a=train["group_a"], bm__group_b=train["group_b"])

            _, metric = self._predict_and_evaluate(pipeline, train, test)

            time_pipeline = time.time() - start_time

            return pd.DataFrame(
                {"dataset": [dataset_name], "mitigator": [mitigator_name], "time": [time_pipeline], "value": [metric]}
            )
        except Exception as e:
            logger.exception(f"Error running mitigator {mitigator_name}: {e!s}")  # noqa: TRY401
            return pd.DataFrame(
                {
                    "dataset": [dataset_name],
                    "mitigator": [mitigator_name],
                    "time": [time.time() - start_time],
                    "value": [None],
                    "error": [str(e)],
                }
            )
