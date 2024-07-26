import logging
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from holisticai.bias.metrics import average_odds_diff, statistical_parity
from holisticai.utils.trade_off_analysers.utils_fairea import (
    do_line_segments_intersect,
    find_segments,
    get_area,
    line,
)

logger = logging.getLogger(__name__)

METRICS = {
    "acc": accuracy_score,
    "auc": roc_auc_score,
    "sp": statistical_parity,
    "aod": average_odds_diff,
}


class Fairea:
    """
    Fairea class for calculating the trade-off between accuracy and fairness of a binary classification model.

    Fairea uses a model behaviour mutation method to create a baseline that can be used to compare quantitatively\
    the fairness-accuracy trade-off for different bias mitigation algorithms and then evaluate their effectiveness\
    in the given scenario. To perform this analysis, the approach consists of three separate steps:

    - A baseline is created by fitting a model without mitigation and then applying a model behaviour\
    mutation to gradually changes the model outputs.
    - Map the given fitted bias mitigation models into five mitigation regions to classify their effectiveness.
    - Assess the effectiveness trade-off by measuring the gap between the mitigators' effectiveness and the baseline.

    Parameters
    ----------
    fair_metric: str
        fairness metric to be used: statistical parity (sp) or average odds difference (aod)
    acc_metric: str
        accuracy metric to be used in the baseline creation and trade-off comparison: accuracy (acc) or roc-auc (auc)
    verbose: bool
        whether to print the progress or not

    Methods
    -------
    create_baseline(x, y, group_a, group_b, test_size, data_splits, repetitions,\
        odds, options, degrees)
        Creates the baseline model for the dataset.
    baseline_metrics()
        Returns the baseline metrics.
    add_model_outcomes(model_name, y_true, y_pred, group_a, group_b)
        Adds the outcomes of a new model to be compared with the baseline model.
    plot_baseline(ax, figsize, title, normalize)
        Plots the baseline model.
    plot_methods(ax, figsize, title, normalize)
        Plots the baseline with the added models.
    trade_off_region_classification()
        Classifies the region of each model.
    compute_trade_off_area()
        Determines the area of each model and saves the best method.
    get_best_model()
        Returns the best model.

    Attributes
    ----------
    verbose: bool
        whether to print the progress or not
    acc_fn: function
        accuracy function to be used
    fair_fn: function
        fairness function to be used
    fair_metric: str
        fairness metric to be used
    methods: dict
        dictionary of the methods added to the class
    normalized_methods: dict
        dictionary of the normalized methods added to the class
    best: str
        best method added to the class
    mitigation_regions: dict
        dictionary of the mitigation regions of the methods added to the class
    odds: dict
        dictionary of odds to be used for mutation
    options: list
        list of options to be used for mutation
    degrees: int
        number of divisions in the range of 0-1 to be used for degrees mutation


    References
    ----------
    ..  [1] Max Hort, Jie M. Zhang, Federica Sarro, and Mark Harman. 2021. Fairea: a model behaviour\
        mutation approach to benchmarking bias mitigation methods. In Proceedings of the 29th ACM\
        Joint Meeting on European Software Engineering Conference and Symposium on the Foundations\
        of Software Engineering (ESEC/FSE 2021)
    """

    def __init__(
        self,
        fair_metric="sp",
        acc_metric="acc",
        verbose=False,  # noqa: FBT002
    ):
        self.verbose = verbose
        self.acc_fn = METRICS[acc_metric]
        self.fair_fn = METRICS[fair_metric]
        self.fair_metric = fair_metric
        self.methods = {}
        self.normalized_methods = {}
        self.best = None
        self.mitigation_regions = {}

    def create_baseline(
        self,
        x,
        y,
        group_a,
        group_b,
        test_size=0.3,
        data_splits=10,
        repetitions=10,
        odds=None,
        options=None,
        degrees=10,
    ):
        """
        Creates the baseline model for the dataset.

        Parameters
        ----------
        x: array, shape=(n_samples, n_features)
            input data
        y: array, shape = (n_samples,)
            target data
        group_a: array, shape = (n_samples,)
            protected attribute for group a
        group_b: array, shape = (n_samples,)
            protected attribute for group b
        test_size: float
            percentage size of the test set (in range 0,1)
        data_splits: int
            number of splits to be used for cross-validation
        repetitions: int
            number of repetitions of mutation to be performed
        odds: dict
            dictionary of odds to be used for mutation
        options: list
            list of options to be used for mutation
        degrees: int
            number of divisions in the range of 0-1 to be used for degrees mutation
        """
        if odds is None:
            self.odds = {"0": [1, 0], "1": [0, 1]}
        if options is None:
            self.options = [0, 1]

        self.degrees = degrees

        # _check_same_shape([group_a, group_b, x, y], names="group_a, group_b, x, y")
        list_of_arr = [group_a, group_b, x, y]
        length = list_of_arr[0].shape[0]

        # check if all arrays have the same length
        if not all(arr.shape[0] == length for arr in list_of_arr):
            msg = "All arrays must have the same length"
            raise ValueError(msg)

        self.baseline_acc, self.baseline_fairness = self.__create_baseline(
            x,
            y,
            group_a,
            group_b,
            test_size=test_size,
            data_splits=data_splits,
            repetitions=repetitions,
        )
        self.acc_scaler = MinMaxScaler()
        self.fair_scaler = MinMaxScaler()
        self.acc_norm = self.acc_scaler.fit_transform(self.baseline_acc).reshape(-1)
        self.fairness_norm = self.fair_scaler.fit_transform(self.baseline_fairness).reshape(-1)

    def baseline_metrics(self):
        """
        Returns the baseline metrics.

        Returns
        -------
        baseline_acc: array, shape = (n_samples,)
            baseline accuracy
        baseline_fairness: array, shape = (n_samples,)
            baseline fairness
        """
        return self.baseline_acc, self.baseline_fairness

    def __mutate_preds(self, preds, to_mutate, ids, odds, options):
        """
        Mutates the predictions of the model.

        Parameters
        ----------
        preds: array, shape = (n_samples,)
            predictions of the model
        to_mutate: int
            number of labels to mutate
        ids: list
            ids of the predictions to be mutated
        odds: list
            odds to be used for mutation
        options: list
            list of options to be used for mutation

        Returns
        -------
        changed: array, shape = (n_samples,)
            mutated predictions
        """
        rand = np.random.choice(options, to_mutate, p=odds)
        # Select prediction ids that are being mutated
        to_change = np.random.choice(ids, size=to_mutate, replace=False)
        changed = np.copy(preds)
        for t, r in zip(to_change, rand):
            changed[t] = r
        return changed

    def __create_baseline(
        self,
        x,
        y,
        group_a,
        group_b,
        test_size,
        data_splits,
        repetitions,
    ):
        """
        Creates the baseline model for the dataset by applying the mutations.

        Parameters
        ----------
        x: array, shape=(n_samples, n_features)
            input data
        y: array, shape = (n_samples,)
            target data
        group_a: array, shape = (n_samples,)
            protected attribute for group a
        group_b: array, shape = (n_samples,)
            protected attribute for group b
        test_size: float
            percentage size of the test set (in range 0,1)
        data_splits: int
            number of splits to be used for cross-validation
        repetitions: int
                    number of repetitions of mutation to be performed

        Returns
        -------
        acc_base: array, shape = (n_samples,)
            baseline accuracy
        """
        n_samples = int(x.shape[0] * test_size)

        ids = list(range(n_samples))

        results = defaultdict(lambda: defaultdict(list))

        for s in range(data_splits):
            if self.verbose:
                logger.info("Current datasplit: {s}")

            np.random.seed(s)
            (
                x_train,
                x_test,
                y_train,
                y_test,
                _,
                group_a_ts,
                _,
                group_b_ts,
            ) = train_test_split(x, y, group_a, group_b, test_size=test_size)
            # Define the pipeline
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])

            # Fit the pipeline and make predictions
            pipe.fit(x_train, y_train)
            pred = pipe.predict(x_test).reshape(-1, 1)

            degrees_ = np.linspace(0, 1, self.degrees + 1)

            # Mutate labels for each degree
            for degree in degrees_:
                # total number of labels to mutate
                to_mutate = int(n_samples * degree)

                for name, o in self.odds.items():
                    # Store each mutation attempt
                    hist = []
                    for _ in range(repetitions):
                        # Generate mutated labels
                        mutated = self.__mutate_preds(pred, to_mutate, ids, o, self.options)
                        # Determine accuracy and fairness of mutated model
                        acc = self.acc_fn(y_test, mutated)
                        if self.fair_metric == "aod":
                            fair = abs(self.fair_fn(group_a_ts, group_b_ts, mutated, y_test))
                        else:
                            fair = abs(self.fair_fn(group_a_ts, group_b_ts, mutated))
                        hist.append([acc, fair])
                    results[name][degree] += hist

        acc_base = np.array([np.mean([row[0] for row in results["0"][degree]]) for degree in degrees_])
        fair_base = np.array([np.mean([row[1] for row in results["0"][degree]]) for degree in degrees_])
        return acc_base.reshape(-1, 1), fair_base.reshape(-1, 1)

    def add_model_outcomes(self, model_name, y_true, y_pred, group_a, group_b):
        """
        Adds the outcomes of a new model to be compared with the baseline model.

        Parameters
        ----------
        model_name: str
                name of the model to be added
        y_true: array, shape = (n_samples,)
                target data
        y_pred: array, shape = (n_samples,)
                predictions of the model
        group_a: array, shape = (n_samples,)
                protected attribute for group a
        group_b: array, shape = (n_samples,)
                protected attribute for group b
        """
        list_of_arr = [group_a, group_b, y_pred, y_true]
        length = list_of_arr[0].shape[0]

        # check if all arrays have the same length
        if not all(arr.shape[0] == length for arr in list_of_arr):
            msg = "All arrays must have the same length"
            raise ValueError(msg)

        acc = self.acc_fn(y_true, y_pred)
        if self.fair_metric == "aod":
            fair = abs(self.fair_fn(group_a, group_b, y_pred, y_true))
        else:
            fair = abs(self.fair_fn(group_a, group_b, y_pred))
        self.methods[model_name] = (fair, acc)
        self.normalized_methods[model_name] = (
            self.fair_scaler.transform(np.array([fair]).reshape(1, -1)).reshape(-1)[0],
            self.acc_scaler.transform(np.array([acc]).reshape(1, -1)).reshape(-1)[0],
        )

    def plot_baseline(self, ax=None, figsize=None, title=None, normalize=False):  # noqa: FBT002
        """
        Plots the baseline model.

        Parameters
        ----------
        cmap: string
            color map to be used for plotting
        ax: matplotlib axis
            axis to be used for plotting
        figsize: tuple
            size of the plot
        title: string
            title of the plot
        normalize: boolean
            whether to normalize the data or not
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.set_xlabel("Fairness")
        ax.set_ylabel("Accuracy")
        if normalize:
            ax.plot(
                self.fairness_norm,
                self.acc_norm,
                marker="|",
                linestyle="solid",
                linewidth=3,
                markersize=16,
                label="Baseline",
            )
        else:
            ax.plot(
                self.baseline_fairness,
                self.baseline_acc,
                marker="|",
                linestyle="solid",
                linewidth=3,
                markersize=16,
                label="Baseline",
            )
        ax.set_xlim(0)
        ax.legend(loc="best")
        return ax

    def plot_methods(self, ax=None, figsize=None, title=None, normalize=False):  # noqa: FBT002
        """
        Plots the baseline with the added models.


        Parameters
        ----------
        cmap: string
            color map to be used for plotting
        ax: matplotlib axis
            axis to be used for plotting
        figsize: tuple
            size of the plot
        title: string
            title of the plot
        normalize: boolean
            whether to normalize the data or not
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.set_xlabel("Fairness")
        ax.set_ylabel("Accuracy")
        if normalize:
            ax.plot(
                self.fairness_norm,
                self.acc_norm,
                marker="|",
                linestyle="solid",
                linewidth=3,
                markersize=16,
                label="Baseline",
            )
        else:
            ax.plot(
                self.baseline_fairness,
                self.baseline_acc,
                marker="|",
                linestyle="solid",
                linewidth=3,
                markersize=16,
                label="Baseline",
            )
        if normalize:
            for name, (fair, acc) in self.normalized_methods.items():
                ax.scatter(fair, acc, color="red")
                ax.annotate(name, (fair, acc))
        else:
            for name, (fair, acc) in self.methods.items():
                ax.scatter(fair, acc, color="red")
                ax.annotate(name, (fair, acc))
        ax.set_xlim(0)
        ax.legend(loc="best")
        return ax


    def determine_region(self, p, fairness_norm, acc_norm):
        """
        Determines the region of the model.

        Parameters
        ----------
        p: tuple
            point of the model
        fairness_norm: list
            normalized fairness of the model
        acc_norm: list
            normalized accuracy of the model

        Returns
        -------
        region: str
            region of the model
        """
        fair_segment, acc_segment = find_segments(p, fairness_norm, acc_norm)
        # Extend bias mitigation point towards four directions (left,right,up,down)
        line_down = line([p[0], p[1]], [p[0], 0])
        line_right = line([p[0], p[1]], [2, p[1]])
        # Determine bias mitigation region based on intersection with baseline
        if p[0] > 1 and p[1] > 1:
            return "inverted"
        if p[0] > 1 and p[1] < 1:
            return "lose-lose"
        if p[0] < 1 and p[1] > 1:
            return "win-win"
        if p[0] >= 0 and p[0] <= 1 and p[1] >= 0 and p[1] <= 1:
            if do_line_segments_intersect(line_down, fair_segment) and do_line_segments_intersect(
                line_right, acc_segment
            ):
                return "good trade-off"
            return "bad trade-off"
        if p[0] < 0:
            return "lose-lose"
        return "inverted"

    def trade_off_region_classification(self, plot=False):  # noqa: FBT002
        """
        Classifies the region of each model.

        Returns
        -------
        df : DataFrame
            dataframe that contains the region of each model of the class
        """
        for k, (fair, acc) in self.normalized_methods.items():
            # define a point for each bias mitigation method
            p = (fair, acc)
            self.mitigation_regions[k] = self.determine_region(p, self.fairness_norm, self.acc_norm)
        # create a dataframe for the region of each method
        df = pd.DataFrame.from_dict(self.mitigation_regions, orient="index", columns=["Region"])
        df.style.set_properties(**{"font-weight": "bold"})
        if plot:
            ax = self.plot_regions(list(zip(self.fairness_norm, self.acc_norm)))
            for name, (fair, acc) in self.normalized_methods.items():
                ax.scatter(fair, acc, color="red")
                ax.annotate(name, (fair, acc))
            return ax
        return df

    def plot_regions(self, polyline):
        # Sort the polyline points by x-coordinate
        polyline.sort(key=lambda p: p[0])

        # Create a figure and a set of subplots
        fig, ax = plt.subplots()

        # Plot the polyline
        polyline_x, polyline_y = zip(*polyline)
        ax.plot(polyline_x, polyline_y, color='black', label='Baseline')

        # Fill the regions
        ax.fill_between(polyline_x, polyline_y, polyline_y[-1], where=[y <= polyline_y[-1] for y in polyline_y],
                        color='green', alpha=0.2, hatch='//', label='Good trade-off')
        ax.fill_between(polyline_x, polyline_y, 0, where=[y >= 0 for y in polyline_y], color='red', alpha=0.2,
                        hatch='\\\\', label='Bad trade-off')
        ax.fill_between(polyline_x, polyline_y[-1], polyline_y[-1]+1, color='blue', alpha=0.2, hatch='--',
                        label='Win-win')
        ax.fill_between(np.arange(polyline_x[-1], polyline_x[-1] + 1, 0.01), 0, polyline_y[-1], color='orange',
                        alpha=0.2, hatch='++', label='Lose-lose')
        ax.fill_between(np.arange(polyline_x[-1], polyline_x[-1]+1, 0.01), polyline_y[-1], polyline_y[-1]+1,
                        color='purple', alpha=0.2, hatch='**', label='Inverted')


        # Set the x and y limits
        ax.set_xlim(0, polyline_x[-1]+1)
        ax.set_ylim(0, polyline_y[-1]+1)

        # x-axis label
        ax.set_xlabel('Normalised Fairness')

        # y-axis label
        ax.set_ylabel('Normalised Accuracy')

        # Place the legend outside the plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return ax

    def compute_trade_off_area(self):
        """
        Determines the area of each model and saves the best method.

        Returns
        -------
        df : DataFrame
            dataframe that contains the area of each model of the class
        """
        good = {k for k, v in self.mitigation_regions.items() if v == "good trade-off"}
        if not good:
            # sys.stdout.write("There are not good methods, consider using a different fairness metric")
            methods = [v for _, v in self.mitigation_regions.items() if v == "win-win"]
            if all(method == "win-win" for method in methods):
                sys.stdout.write("You can choose one of the win-win methods")
            else:
                sys.stdout.write("There are not good methods, consider using a different fairness metric")
            return
        area_methods = {}
        for item in good:
            mitigator_point = self.normalized_methods[item]
            area = get_area(mitigator_point, self.fairness_norm, self.acc_norm)
            area_methods[item] = area
        # create a dataframe for the area of each method
        df = pd.DataFrame.from_dict(area_methods, orient="index", columns=["Area"])
        # sort the dataframe based on the Area
        df = df.sort_values(by="Area", ascending=False)
        df.style.set_properties(**{"font-weight": "bold"})
        # save method with the largest area in the best variable
        self.best = df.index[0]
        return df

    def get_best_model(self):
        """
        Returns the best model.

        Returns
        -------
        best: str
            best method added to the class
        """
        if self.best is None:
            methods = [v for _, v in self.mitigation_regions.items() if v == "win-win"]
            if all(method == "win-win" for method in methods):
                logger.warning("You can choose one of the win-win methods")
            else:
                logger.warning("There are not good methods, consider using a different fairness metric")
        return self.best
