import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
from tqdm import tqdm

from holisticai.benchmark.utils import load_benchmark
from holisticai.bias.metrics import classification_bias_metrics
from holisticai.datasets import load_dataset
from holisticai.pipeline import Pipeline
from holisticai.utils._plotting import get_colors

DATASETS = [
    "adult",
    "law_school",
    "german_credit",
    "census_kdd",
    "bank_marketing",
    "credit_card",
    "compas_recidivism",
    "diabetes",
]

MODELS = [LogisticRegression(), RandomForestClassifier(), GradientBoostingClassifier()]


class BinaryClassificationBenchmark:
    def __init__(self):
        pass

    def __str__(self):
        return "BinaryClassificationBenchmark"

    def run_benchmark(self, mitigator=None, type=None):
        self.mitigator_name = mitigator.__class__.__name__

        if mitigator is None:
            raise ValueError("Please provide a mitigator to run the benchmark")
        if type is None:
            raise ValueError(
                "Please provide a type: preprocessing, inprocessing or postprocessing"
            )

        print(f"Binary Classification Benchmark initialized for {self.mitigator_name}")

        results_dataframe = pd.DataFrame()
        bench_dataframe = load_benchmark(
            task="binary_classification", type=type
        ).reset_index(drop=True)

        for data_name in tqdm(DATASETS):
            for model in MODELS:
                df, group_a, group_b = load_dataset(
                    dataset=data_name, preprocessed=True, as_array=False
                )
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    group_a_tr,
                    group_a_ts,
                    group_b_tr,
                    group_b_ts,
                ) = train_test_split(
                    X, y, group_a, group_b, test_size=0.2, random_state=42
                )
                train_data = X_train, y_train, group_a_tr, group_b_tr
                test_data = X_test, y_test, group_a_ts, group_b_ts

                if type == "preprocessing":
                    pipeline = Pipeline(
                        steps=[
                            ("scalar", StandardScaler()),
                            ("bm_preprocessing", mitigator),
                            ("model", model),
                        ]
                    )

                elif type == "inprocessing":
                    pipeline = Pipeline(
                        steps=[
                            ("scalar", StandardScaler()),
                            (
                                "bm_inprocessing",
                                mitigator.transform_estimator(feature_dim=X.shape[1]),
                            ),
                        ]
                    )

                elif type == "postprocessing":
                    pipeline = Pipeline(
                        steps=[
                            ("scalar", StandardScaler()),
                            ("model", model),
                            ("bm_postprocessing", mitigator),
                        ]
                    )

                X_train, y_train, group_a_train, group_b_train = train_data
                X_test, y_test, group_a_test, group_b_test = test_data

                fit_params = {
                    "bm__group_a": group_a_train,
                    "bm__group_b": group_b_train,
                }

                pipeline.fit(X_train, y_train, **fit_params)

                predict_params = {
                    "bm__group_a": group_a_test,
                    "bm__group_b": group_b_test,
                }

                y_pred = pipeline.predict(X_test, **predict_params)
                metrics = classification_bias_metrics(
                    group_a_test, group_b_test, y_pred, y_test, metric_type="both"
                )
                metrics_result = (
                    metrics.copy()
                    .drop(columns="Reference")
                    .rename(columns={"Value": f"{mitigator.__class__.__name__}"})
                    .T
                )
                metrics_result.insert(0, "Dataset", data_name)
                metrics_result.insert(1, "Model", model.__class__.__name__)
                metrics_result = metrics_result.reset_index(drop=False).rename(
                    columns={"index": "Mitigator"}
                )
                results_dataframe = pd.concat(
                    [results_dataframe, metrics_result], axis=0
                )

        self.results = results_dataframe.reset_index(drop=True)
        self.results_benchmark = (
            pd.concat([self.results, bench_dataframe], axis=0)
            .sort_values(by=["Dataset", "Statistical Parity"], ascending=False)
            .reset_index(drop=True)
        )
        ranking = abs(
            self.results_benchmark.pivot_table(
                index="Mitigator",
                columns="Dataset",
                values="Statistical Parity",
                aggfunc="mean",
            )
        )
        ranking.insert(0, "Mean", ranking.mean(axis=1))
        self.results_ranking = ranking.sort_values(by="Mean", ascending=True)

    def highlight_line(self, s):
        return [
            "background-color: mediumslateblue" if v == self.mitigator_name else ""
            for v in s
        ]

    def evaluate_table(self, ranking=True, highlight=True, tab=False):
        benchmark_table = self.results if not ranking else self.results_ranking
        print(
            "Benchmark Table: the values are |frac{1}{M}\sum_{i}^{M}(StatiticalParity_{i}))| by dataset, \n where M is the number of models trained in benchmark."
        )
        if highlight:

            def highlight_mitigator_name(val):
                return [
                    "background: mediumslateblue"
                    if val.name == self.mitigator_name
                    else ""
                    for _ in val
                ]

            return benchmark_table.style.apply(highlight_mitigator_name, axis=1)

        elif tab:
            print(
                tabulate(
                    benchmark_table, headers=benchmark_table.columns, tablefmt="pretty"
                )
            )

        else:
            return benchmark_table

    def evaluate_plot(self, metric="Statistical Parity", benchmark=True):

        data = self.results_benchmark
        if benchmark is not True:
            data = self.results

        colors = get_colors(len(data["Model"].unique()))
        hai_palette = sns.color_palette(colors)

        for i in range(len(data["Dataset"].unique())):
            fig = plt.figure(figsize=(15, 5))
            level_1 = data["Dataset"].unique()[i]
            temp = data[(data["Dataset"] == level_1)]
            sns.barplot(
                y=metric,
                x="Mitigator",
                hue="Model",
                palette=hai_palette,
                data=temp,
            )
            plt.axhline(0, color="red", linewidth=2, linestyle="--")
            plt.grid()
            plt.xlabel("")
            plt.xticks(rotation=45)
            plt.title(f"Dataset: {level_1}")
            plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
            plt.tight_layout()
            plt.show()

    def benchmark(self, type=None):
        if type is None:
            raise ValueError(
                "Please provide a type: preprocessing, inprocessing or postprocessing"
            )
        return load_benchmark(task="binary_classification", type=type, ranking=True)

    def submit(self):
        name = self.mitigator_name
        print(f"{name} benchmark submitted")
        print(f"{name} benchmark submitted")

        print("https://holistic-ai.com/benchmark/binary_classification")
