import warnings

warnings.filterwarnings("ignore")

import webbrowser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from tabulate import tabulate
from tqdm import tqdm

from holisticai.benchmark.utils import load_benchmark
from holisticai.bias.metrics import rmse_fairness_score, regression_bias_metrics
from holisticai.datasets import load_dataset
from holisticai.pipeline import Pipeline
from holisticai.utils._plotting import get_colors
from holisticai.utils.transformers.bias import BMInprocessing as BMImp
from holisticai.utils.transformers.bias import BMPostprocessing as BMPost
from holisticai.utils.transformers.bias import BMPreprocessing as BMPre

DATASETS = [
   "crime",
]

MODELS = [LinearRegression(), RandomForestRegressor(), GradientBoostingRegressor()]


class RegressionBenchmark:
    def __init__(self):
        pass

    def __str__(self):
        return "RegressionBenchmark"

    def run_benchmark(self, custom_mitigator=None, type=None, _implemented=False):
        if custom_mitigator is None:
            raise ValueError("Please provide a mitigator to run the benchmark")
        
        if type is None:
            raise ValueError(
                "Please provide a type: preprocessing, inprocessing or postprocessing"
            )
        
        if _implemented:
            self.custom_mitigator = custom_mitigator
            self.mitigator_name = custom_mitigator.__class__.__name__

        else:
            if type == "preprocessing":
                class BenchmarkConfig(BaseEstimator, BMPre):

                    def __init__(self, mitigator=None):
                        self.mitigator = mitigator

                    def fit(self, X, group_a, group_b):
                        params = self._load_data(X=X, group_a=group_a, group_b=group_b)
                        X = params["X"]
                        group_a = params["group_a"]
                        group_b = params["group_b"]

                        self.model_ = self.mitigator.fit(X, group_a, group_b)
                        return self

                    def transform(self, X, group_a, group_b):
                        return self.model_.transform(X, group_a, group_b)

            elif type == "inprocessing":
                class BenchmarkConfig(BaseEstimator, BMImp):

                    def __init__(self, mitigator=None):
                        self.mitigator = mitigator

                    def transform_estimator(self, estimator=None):
                        if estimator is not None:
                            self.estimator = estimator
                        else:
                            self.estimator = clone(self.estimator)
                        return self

                    def fit(self, X, y_true, group_a, group_b):
                        self.estimator_ = clone(self.estimator)

                        model = self.mitigator.fit(X, y_true, group_a, group_b, self.estimator_)
                        self.model_ = model

                        return self

                    def predict(self, X):
                        return self.model_.predict(X)

                    def predict_proba(self, X):
                        return self.model_.predict_proba(X)
            
            elif type == "postprocessing":
                class BenchmarkConfig(BaseEstimator, BMPost):

                    def __init__(self, mitigator=None):
                        self.mitigator = mitigator

                    def fit(self, y_pred, group_a, group_b):
                        params = self._load_data(y_pred=y_pred, group_a=group_a, group_b=group_b)
                        group_a = params["group_a"] == 1
                        group_b = params["group_b"] == 1
                        y_pred = params["y_pred"]

                        self.model_ = self.mitigator.fit(y_pred, group_a, group_b)
                        return self

                    def transform(self, y_pred, group_a, group_b):
                        params = self._load_data(y_pred=y_pred, group_a=group_a, group_b=group_b)
                        group_a = params["group_a"] == 1
                        group_b = params["group_b"] == 1
                        y_pred = params["y_pred"]

                        result = self.model_.transform(y_pred, group_a, group_b)
                        return result
                
            self.mitigator = BenchmarkConfig(mitigator=custom_mitigator)
            self.mitigator_name = custom_mitigator.__class__.__name__

        print(f"Regression Benchmark initialized for {self.mitigator_name}")

        results_dataframe = pd.DataFrame()
        bench_dataframe = load_benchmark(
            task="regression", type=type
        ).reset_index(drop=True)

        for data_name in tqdm(DATASETS):
            for model in MODELS:
                np.random.seed(10)
                df, group_a, group_b = load_dataset(
                    dataset=data_name, preprocessed=True, as_array=False
                )
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values.astype(int)
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
                            ("bm_preprocessing", self.mitigator),
                            ("model", model),
                        ]
                    )

                elif type == "inprocessing":
                    pipeline = Pipeline(
                        steps=[
                            ("scalar", StandardScaler()),
                            (
                                "bm_inprocessing",
                                self.mitigator.transform_estimator(estimator=model),
                            ),
                        ]
                    )

                elif type == "postprocessing":
                    pipeline = Pipeline(
                        steps=[
                            ("scalar", StandardScaler()),
                            ("model", model),
                            ("bm_postprocessing", self.mitigator),
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
                mse = mean_squared_error(y_test, y_pred)
                rfs = rmse_fairness_score(
                    group_a_test, group_b_test, y_pred, y_test
                )
                metrics = regression_bias_metrics(
                    group_a_test, group_b_test, y_pred, y_test, metric_type="both"
                )
                metrics_result = (
                    metrics.copy()
                    .drop(columns="Reference")
                    .rename(columns={"Value": f"{self.mitigator_name}"})
                    .T
                )
                metrics_result.insert(0, "Dataset", data_name)
                metrics_result.insert(1, "Model", model.__class__.__name__)
                metrics_result.insert(2, "RFS", rfs)
                metrics_result.insert(3, "MSE", mse)
                metrics_result = metrics_result.reset_index(drop=False).rename(
                    columns={"index": "Mitigator"}
                )
                results_dataframe = pd.concat(
                    [results_dataframe, metrics_result], axis=0
                )

        self.results = results_dataframe.reset_index(drop=True)
        self.results_benchmark = (
            pd.concat([self.results, bench_dataframe], axis=0)
            .sort_values(by=["Dataset", "RFS"], ascending=False)
            .reset_index(drop=True)
        )
        ranking = abs(
            self.results_benchmark.pivot_table(
                index="Mitigator",
                columns="Dataset",
                values="RFS",
                aggfunc="mean",
            )
        )
        ranking.insert(0, "Average RFS", ranking.mean(axis=1))
        ranking.insert(1, "Average MSE", self.results.groupby("Mitigator")["MSE"].mean())
        self.results_ranking = ranking.sort_values(by="Average RFS", ascending=False)

    def highlight_line(self, s):
        return [
            "background-color: mediumslateblue" if v == self.mitigator_name else ""
            for v in s
        ]

    def evaluate_table(self, ranking=True, highlight=True, tab=False):
        benchmark_table = self.results if not ranking else self.results_ranking
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

    def evaluate_plot(self, metric="RFS", benchmark=True):

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

    def benchmark(self, type=None, ranking=True):
        if type is None:
            raise ValueError(
                "Please provide a type: preprocessing, inprocessing or postprocessing"
            )
        return load_benchmark(task="regression", type=type, ranking=ranking)

    def submit(self):
        link = "https://forms.office.com/r/Vd6FT4eNL2"
        print("Opening the link in your browser: https://forms.office.com/r/Vd6FT4eNL2")
        webbrowser.open(link, new=2)
