import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from holisticai.datasets import load_adult, load_last_fm

from .tests_data_utils import (
    post_process_dataframe,
    post_process_dataset,
    post_process_recommender,
    preprocess_adult_dataset,
    preprocess_heart_dataset,
    preprocess_lastfm_dataset,
    preprocess_us_crime_dataset,
    remove_nans,
    sample_adult,
    sample_categorical,
    sample_heart,
)

# dictionnary of metrics
metrics_dict = {
    "Accuracy": metrics.accuracy_score,
    "Balanced accuracy": metrics.balanced_accuracy_score,
    "Precision": metrics.precision_score,
    "Recall": metrics.recall_score,
    "F1-Score": metrics.f1_score,
}

# efficacy metrics dataframe helper tool
def metrics_dataframe(y_pred, y_true, metrics_dict=metrics_dict):
    metric_list = [[pf, fn(y_true, y_pred)] for pf, fn in metrics_dict.items()]
    return pd.DataFrame(metric_list, columns=["Metric", "Value"]).set_index("Metric")


def format_result_colum(name, config):
    return config["result"].rename(columns={"Value": name}).iloc[:, 0]


def show_result_table(configurations, df_baseline):
    table = pd.concat(
        [df_baseline.iloc[:, 0]]
        + [format_result_colum(name, config) for name, config in configurations.items()]
        + [df_baseline.iloc[:, 1]],
        axis=1,
    )
    return table.rename(columns={"Value": "Baseline"})


def check_results(df1, df2):
    print(f"Equal: {df1.equals(df2)}")
    df = pd.concat([df1["Value"], df2[["Value", "Reference"]]], axis=1)
    df.columns = ["without pipeline", "with pipeline", "Reference"]
    print(df)
    return df


class Dclass:
    def __init__(self):
        self.output_label = "O_0"
        self.favorable_label = 1
        self.unfavorable_label = 0

    def load_preprocessed_adult_df(self):
        dataset = load_adult()
        df = pd.concat([dataset["data"], dataset["target"]], axis=1)

        protected_variables = ["sex", "race"]
        output_variable = ["class"]

        y_df = df[output_variable]
        x_df = df.drop(protected_variables + output_variable, axis=1)

        group = ["sex"]
        group_a = pd.DataFrame(df[group] == "Female")
        group_a.columns = ["Female"]
        group_b = pd.DataFrame(df[group] == "Male")
        group_b.columns = ["Male"]
        return pd.concat([x_df, y_df, group_a, group_b], axis=1)

    def get_distortion_adult(self, vold, vnew):
        """Distortion function for the adult dataset. We set the distortion
        metric here. See section 4.3 in supplementary material of
        http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
        for an example
        Note:
            Users can use this as templates to create other distortion functions.
        Args:
            vold (dict) : {attr:value} with old values
            vnew (dict) : dictionary of the form {attr:value} with new values
        Returns:
            d (value) : distortion value
        """

        # Define local functions to adjust education and age
        def adjustEdu(v):
            if v == ">12":
                return 13
            elif v == "<6":
                return 5
            else:
                return int(v)

        def adjustAge(a):
            if a == ">=70":
                return 70.0
            else:
                return float(a)

        def adjustInc(a):
            if a == "<=50K":
                return 0
            elif a == ">50K":
                return 1
            else:
                return int(a)

        # value that will be returned for events that should not occur
        bad_val = 3.0

        # Adjust education years
        eOld = adjustEdu(vold[self.xfm["Education Years"]])
        eNew = adjustEdu(vnew[self.xfm["Education Years"]])

        # Education cannot be lowered or increased in more than 1 year
        if (eNew < eOld) | (eNew > eOld + 1):
            return bad_val

        # adjust age
        aOld = adjustAge(vold[self.xfm["Age (decade)"]])
        aNew = adjustAge(vnew[self.xfm["Age (decade)"]])

        # Age cannot be increased or decreased in more than a decade
        if np.abs(aOld - aNew) > 10.0:
            return bad_val

        # Penalty of 2 if age is decreased or increased
        if np.abs(aOld - aNew) > 0:
            return 2.0

        # Adjust income
        incOld = adjustInc(vold[self.output_label])
        incNew = adjustInc(vnew[self.output_label])

        # final penalty according to income
        if incOld > incNew:
            return 1.0
        else:
            return 0.0

    def custom_preprocessing(self, df, sub_samp=False, balance=False):
        """The custom pre-processing function is adapted from
        https://github.com/fair-preprocessing/nips2017/blob/master/Adult/code/Generate_Adult_Data.ipynb
        If sub_samp != False, then return smaller version of dataset truncated to tiny_test data points.
        """

        # Group age by decade
        df["Age (decade)"] = df["age"].apply(lambda x: x // 10 * 10)
        # df['Age (decade)'] = df['age'].apply(lambda x: np.floor(x/10.0)*10.0)

        def group_edu(x):
            if x <= 5:
                return "<6"
            elif x >= 13:
                return ">12"
            else:
                return x

        def age_cut(x):
            if x >= 70:
                return ">=70"
            else:
                return x

        def group_race(x):
            if x == "White":
                return 1.0
            else:
                return 0.0

        # Cluster education and age attributes.
        # Limit education range
        df["Education Years"] = df["education-num"].apply(lambda x: group_edu(x))
        df["Education Years"] = df["Education Years"].astype("category")

        # Limit age range
        df["Age (decade)"] = df["Age (decade)"].apply(lambda x: age_cut(x))

        if sub_samp and not balance:
            df = df.sample(sub_samp)
        if sub_samp and balance:
            df_0 = df[df["class"] == "<=50K"]
            df_1 = df[df["class"] == ">50K"]
            df_0 = df_0.sample(int(sub_samp / 2))
            df_1 = df_1.sample(int(sub_samp / 2))
            df = pd.concat([df_0, df_1])

        y = df["class"].replace(
            {">50K": self.favorable_label, "<=50K": self.unfavorable_label}
        )
        x = df[["Age (decade)", "Education Years"]]
        group_a = df["Female"]
        group_b = df["Male"]

        self.xfm = {f: f"I_{i}" for i, f in enumerate(x.columns)}

        data = [x, y, group_a, group_b]

        dataset = train_test_split(*data, test_size=0.6, shuffle=True)
        train_data = dataset[::2]
        test_data = dataset[1::2]
        return train_data, test_data


import pytest
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from testing_utils.tests_data_utils import convert_float_to_categorical

from holisticai.datasets import load_us_crime


class MetricsHelper:
    @staticmethod
    def false_negative_rate_difference(group_a, group_b, y_pred, y_true):
        tnra, fpra, fnra, tpra = confusion_matrix(
            y_true[group_a == 1], y_pred[group_a == 1], normalize="true"
        ).ravel()
        tnrb, fprb, fnrb, tprb = confusion_matrix(
            y_true[group_b == 1], y_pred[group_b == 1], normalize="true"
        ).ravel()
        return fnra - fnrb

    @staticmethod
    def true_positive_rate_difference(group_a, group_b, y_pred, y_true):
        tnra, fpra, fnra, tpra = confusion_matrix(
            y_true[group_a == 1], y_pred[group_a == 1], normalize="true"
        ).ravel()
        tnrb, fprb, fnrb, tprb = confusion_matrix(
            y_true[group_b == 1], y_pred[group_b == 1], normalize="true"
        ).ravel()
        return tprb - tpra


@pytest.fixture
def small_categorical_dataset():
    return load_test_dataset("binary", "small")


@pytest.fixture
def small_multiclass_dataset():
    return load_test_dataset("multiclass", "small")


@pytest.fixture
def small_regression_dataset():
    return load_test_dataset("regression", "small")


def fit(model, small_categorical_dataset):
    train_data, test_data = small_categorical_dataset
    X, y, group_a, group_b = train_data

    fit_params = {"bm__group_a": group_a, "bm__group_b": group_b}
    model.fit(X, y, **fit_params)
    return model


def evaluate_pipeline(pipeline, small_categorical_dataset, metric_names, thresholds):
    from holisticai.bias import metrics

    train_data, test_data = small_categorical_dataset
    X, y, group_a, group_b = train_data
    predict_params = {"bm__group_a": group_a, "bm__group_b": group_b}
    y_pred = pipeline.predict(X, **predict_params)

    for metric_name, threshold in zip(metric_names, thresholds):
        if metric_name == "False Negative Rate difference":
            assert (
                MetricsHelper.false_negative_rate_difference(
                    group_a, group_b, y_pred, y
                )
                < threshold
            )
        elif metric_name == "False Positive Rate difference":
            print(metrics.false_positive_rate_diff(group_a, group_b, y_pred, y))
            assert (
                metrics.false_positive_rate_diff(group_a, group_b, y_pred, y)
                < threshold
            )
        elif metric_name == "Statistical parity difference":
            assert abs(metrics.statistical_parity(group_a, group_b, y_pred)) < threshold
        elif metric_name == "Average odds difference":
            assert metrics.average_odds_diff(group_a, group_b, y_pred, y) < threshold
        elif metric_name == "Equal opportunity difference":
            assert (
                MetricsHelper.true_positive_rate_difference(group_a, group_b, y_pred, y)
                < threshold
            )
        else:
            raise Exception(f"Unknown metric {metric_name}")


def process_binary_dataset(size="small"):
    """
    Processes the adult dataset and returns the data, output variable, protected group A and protected group B as numerical arrays

    Parameters
    ----------
    size : str
        The size of the dataset to return. Either 'small' or 'large'

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    data = load_adult()
    protected_attribute = "sex"
    output_variable = "class"
    drop_columns = ["education", "race", "sex", "class"]
    df = pd.concat([data["data"], data["target"]], axis=1)
    df = remove_nans(df)
    if size == "small":
        df = sample_adult(df)
    df, group_a, group_b = preprocess_adult_dataset(
        df, protected_attribute, output_variable, drop_columns
    )
    return post_process_dataset(df, output_variable, group_a, group_b)


def process_regression_dataset(size="small", return_df=False):
    """
    Processes the US crime dataset and returns the data, output variable, protected group A and protected group B as numerical arrays or as dataframe if needed

    Parameters
    ----------
    size : str
        The size of the dataset to return. Either 'small' or 'large'
    return_df : bool
        Whether to return the data as dataframe or not

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    data = load_us_crime()
    protected_attribute = "racePctWhite"
    output_variable = "ViolentCrimesPerPop"
    df = pd.concat([data["data"], data["target"]], axis=1)
    df = remove_nans(df)
    df, group_a, group_b = preprocess_us_crime_dataset(
        df, protected_attribute=protected_attribute
    )
    if size == "small":
        a_index = list(np.where(group_a == 1)[0][::5])
        b_index = list(np.where(group_b == 1)[0][::5])
        df = df.iloc[a_index + b_index]
        group_a = group_a.iloc[a_index + b_index]
        group_b = group_b.iloc[a_index + b_index]
    if return_df:
        return post_process_dataframe(df, group_a, group_b)
    df, group_a, group_b = post_process_dataframe(df, group_a, group_b)
    return post_process_dataset(df, output_variable, group_a, group_b)


def process_clustering_dataset(size="small"):
    """
    Processes the heart dataset and returns the data, output variable, protected group A and protected group B as numerical arrays

    Parameters
    ----------
    size : str
        The size of the dataset to return. Either 'small' or 'large'

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv"
    )
    protected_attribute = "sex"
    output_variable = "DEATH_EVENT"
    drop_columns = ["age", "sex", "DEATH_EVENT"]
    df = remove_nans(df)
    if size == "small":
        df = sample_heart(df)
    df, group_a, group_b = preprocess_heart_dataset(
        df, protected_attribute, output_variable, drop_columns
    )
    return post_process_dataset(df, output_variable, group_a, group_b)


def process_multiclass_dataset(size="small"):
    """
    Processes the US crime dataset as a multiclass dataset and returns the data, output variable, protected group A and protected group B as numerical arrays

    Parameters
    ----------
    size : str
        The size of the dataset to return. Either 'small' or 'large'

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B
    """
    output_variable = "ViolentCrimesPerPop"
    df, group_a, group_b = process_regression_dataset("large", return_df=True)
    if size == "small":
        df, group_a, group_b = sample_categorical(df, group_a, group_b, output_variable)
        df, group_a, group_b = post_process_dataframe(df, group_a, group_b)
        return post_process_dataset(df, output_variable, group_a, group_b)
    print(df.shape)
    df[output_variable] = convert_float_to_categorical(df[output_variable], 5)
    df, group_a, group_b = post_process_dataframe(df, group_a, group_b)
    return post_process_dataset(df, output_variable, group_a, group_b)


def process_recommender_dataset(size="small"):
    """
    Processes the lastfm dataset and returns the data, output variable, protected group A and protected group B as numerical arrays

    Parameters
    ----------
    size : str
        The size of the dataset to return. Either 'small' or 'large'

    Returns
    -------
    data_matrix : np.ndarray
        The numerical pivot array
    p_attr : np.ndarray
        The protected attribute
    """
    bunch = load_last_fm()
    df = bunch["frame"]
    protected_attribute = "sex"
    user_column = "user"
    item_column = "artist"
    df_pivot, p_attr = preprocess_lastfm_dataset(
        df,
        protected_attribute=protected_attribute,
        user_column=user_column,
        item_column=item_column,
    )
    if size == "small":
        a_index = list(np.where(p_attr == 1)[0][:10])
        b_index = list(np.where(p_attr == 0)[0][:10])

        df_pivot = df_pivot.iloc[a_index + b_index]
        p_attr = p_attr[a_index + b_index]
    data_matrix = post_process_recommender(df_pivot)
    return data_matrix, p_attr


def load_test_dataset(dataset="binary", size="small"):
    """
    Loads a test dataset

    Parameters
    ----------
    dataset : str
        The name of the dataset to load
    size : str
        The size of the dataset to load

    Returns
    -------
    tuple
        A tuple with two lists containing the data, output variable, protected group A and protected group B for train and test respectively
    """
    if dataset == "binary":
        return process_binary_dataset(size=size)
    elif dataset == "regression":
        return process_regression_dataset(size=size)
    elif dataset == "clustering":
        return process_clustering_dataset(size=size)
    elif dataset == "multiclass":
        return process_multiclass_dataset(size=size)
    elif dataset == "recommender":
        return process_recommender_dataset(size=size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
