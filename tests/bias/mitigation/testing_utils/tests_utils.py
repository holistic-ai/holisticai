import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

from holisticai.datasets import load_adult

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


def load_preprocessed_adult():
    dataset= load_adult()
    df = pd.concat([dataset["data"], dataset["target"]], axis=1)
    protected_variables = ["sex", "race"]
    output_variable = ["class"]
    favorable_label = 1
    unfavorable_label = 0

    y = df[output_variable].replace(
        {">50K": favorable_label, "<=50K": unfavorable_label}
    )
    x = pd.get_dummies(df.drop(protected_variables + output_variable, axis=1))

    group = ["sex"]
    group_a = df[group] == "Female"
    group_b = df[group] == "Male"
    index_a = list(np.where(group_a == 1)[0])
    index_b = list(np.where(group_b == 1)[0])
    index = index_a[:800] + index_b[:800]
    data = [x.iloc[index], y.iloc[index], group_a.iloc[index], group_b.iloc[index]]

    dataset = train_test_split(*data, test_size=0.5, shuffle=True)
    train_data = dataset[::2]
    test_data = dataset[1::2]
    return train_data, test_data


def load_preprocessed_adult_v2():
    dataset = load_adult()
    df = pd.concat([dataset["data"], dataset["target"]], axis=1)
    df = df.sample(n=600)

    protected_variables = ["sex", "race"]
    output_variable = ["class"]
    favorable_label = 1
    unfavorable_label = 0

    y = df[output_variable].replace(
        {">50K": favorable_label, "<=50K": unfavorable_label}
    )
    features_to_keep = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    x = pd.get_dummies(df[features_to_keep])

    group = ["sex"]
    group_a = df[group] == "Female"
    group_b = df[group] == "Male"
    data = [x, y, group_a, group_b]

    dataset = train_test_split(*data, test_size=0.6, shuffle=True)
    train_data = dataset[::2]
    test_data = dataset[1::2]
    return train_data, test_data


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
from holisticai.datasets import load_us_crime
from testing_utils.tests_data_utils import convert_float_to_categorical

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
    protected_variables = ["sex", "race"]
    output_variable = ["class"]
    favorable_label = 1
    unfavorable_label = 0
    group = ["sex"]
        
    dataset = load_adult()
    df = pd.concat([dataset["data"], dataset["target"]], axis=1)
    df = pd.concat([
                    df[(df[group[0]]=="Male") & (df[output_variable[0]]==">50K")].sample(5).reset_index(drop=True),
                    df[(df[group[0]]=="Male") & (df[output_variable[0]]=="<=50K")].sample(10).reset_index(drop=True),
                    df[(df[group[0]]=="Female") & (df[output_variable[0]]==">50K")].sample(2).reset_index(drop=True),
                    df[(df[group[0]]=="Female") & (df[output_variable[0]]=="<=50K")].sample(5).reset_index(drop=True)], axis=0)
    
    y = df[output_variable].replace({">50K": favorable_label, "<=50K": unfavorable_label}).values.ravel()
    x = pd.get_dummies(df.drop(protected_variables + output_variable, axis=1))
    
    groups = [df[group] == "Female", df[group] == "Male"]
    data = [x, y] + [group.values.ravel() for group in groups]

    train_data = test_data = data
    return train_data,test_data

@pytest.fixture
def small_multiclass_dataset():
    nb_classes = 3
    dataset = load_us_crime()
    df = pd.concat([dataset["data"], dataset["target"]], axis=1)
    
    df_clean = df.iloc[:, [i for i, n in enumerate(df.isna().sum(axis=0).T.values) if n < 1000]].dropna()
    group_a = df_clean["racePctWhite"] > 0.5
    group_b = ~group_a
    xor_groups = group_a ^ group_b

    cols = [c for c in df_clean.columns if not (c.startswith("race") or c.startswith("age"))]
    df_clean = df_clean[cols].iloc[:, 3:].loc[xor_groups]
    group_a, group_b = group_a[xor_groups].reset_index(drop=True), group_b[xor_groups].reset_index(drop=True)

    scalar = StandardScaler()
    df_t = scalar.fit_transform(df_clean)
    X = df_t[:, :-1]
    y = df_t[:, -1] if nb_classes is None else convert_float_to_categorical(df_clean.iloc[:, -1], nb_classes)

    data = []
    for m in [X, y, group_a, group_b]:
        x = pd.DataFrame(m.copy())
        x = pd.concat([
                x[(group_a==1) & (y==0)].iloc[:2],
                x[(group_a==1) & (y==1)].iloc[:2],
                x[(group_a==1) & (y==2)].iloc[:2],
                x[(group_b==1) & (y==0)].iloc[:2],
                x[(group_b==1) & (y==1)].iloc[:2],
                x[(group_b==1) & (y==2)].iloc[:2],
                ], axis=0).reset_index(drop=True)
        data.append(x)
    data = [data[0], data[1].values.ravel(), data[2].values.ravel(), data[3].values.ravel()]   
    return data, data

@pytest.fixture
def small_regression_dataset():
    dataset = load_us_crime()
    df = pd.concat([dataset["data"], dataset["target"]], axis=1)    
    df_clean = df.iloc[:, [i for i, n in enumerate(df.isna().sum(axis=0).T.values) if n < 1000]].dropna()
    group_a = df_clean["racePctWhite"] > 0.5
    group_b = ~group_a
    xor_groups = group_a ^ group_b

    cols = [c for c in df_clean.columns if not (c.startswith("race") or c.startswith("age"))]
    df_clean = df_clean[cols].iloc[:, 3:].loc[xor_groups]
    group_a, group_b = group_a[xor_groups], group_b[xor_groups]

    scalar = StandardScaler()
    df_t = scalar.fit_transform(df_clean)
    X = np.array(df_t[:, :-1])
    y = np.array(df_t[:, -1])
    a_index = list(np.where(group_a==1)[0][:5])
    b_index = list(np.where(group_b==1)[0][:5])
    indexes = (a_index + b_index)
    X = np.stack([X[i] for i in indexes],axis=0)
    y = np.array([y[i] for i in indexes])
    group_a = np.array([group_a.values[i] for i in indexes])
    group_b = np.array([group_b.values[i] for i in indexes])
    train_data = test_data = X,y,group_a,group_b
    return train_data , test_data

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
