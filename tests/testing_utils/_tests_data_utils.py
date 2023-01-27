import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from holisticai.datasets import load_adult, load_us_crime


def load_preprocessed_adult(short_version=True):
    dataset = load_adult()
    df = pd.concat([dataset["data"], dataset["target"]], axis=1)
    if not short_version:
        df = df.sample(n=500)

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
    data = [x, y, group_a, group_b]

    dataset = train_test_split(*data, test_size=0.2, shuffle=True)
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


def convert_float_to_categorical(target, nb_classes, numeric_classes=True):
    eps = np.finfo(float).eps
    if numeric_classes:
        labels = list(range(nb_classes))
    else:
        labels = [f"Q{c}-Q{c+1}" for c in range(nb_classes)]
    labels_values = np.linspace(0, 1, nb_classes + 1)
    v = np.array(target.quantile(labels_values)).squeeze()
    v[0], v[-1] = v[0] - eps, v[-1] + eps
    y = target.copy()
    for (i, c) in enumerate(labels):
        y[(target.values >= v[i]) & (target.values < v[i + 1])] = c
    return y.astype(np.int32)


def load_preprocessed_us_crime(nb_classes=None):
    from sklearn.preprocessing import StandardScaler

    dataset = load_us_crime()

    df = pd.concat([dataset["data"], dataset["target"]], axis=1)
    df_clean = df.iloc[
        :, [i for i, n in enumerate(df.isna().sum(axis=0).T.values) if n < 1000]
    ]
    df_clean = df_clean.dropna()

    group_a = df_clean["racePctWhite"].apply(lambda x: x > 0.5)
    group_b = 1 - group_a
    xor_groups = group_a ^ group_b

    # Remove sensitive groups from dataset
    cols = [
        c
        for c in df_clean.columns
        if (not c.startswith("race")) and (not c.startswith("age"))
    ]
    df_clean = df_clean[cols].iloc[:, 3:]
    df_clean = df_clean[xor_groups]
    group_a = group_a[xor_groups]
    group_b = group_b[xor_groups]

    scalar = StandardScaler()
    df_t = scalar.fit_transform(df_clean)
    X = df_t[:, :-1]

    if nb_classes is None:
        y = df_t[:, -1]
    else:
        y = convert_float_to_categorical(df_clean.iloc[:, -1], nb_classes=nb_classes)

    data = X, y, group_a, group_b
    datasets = train_test_split(*data, test_size=0.2)
    train_data = datasets[::2]
    test_data = datasets[1::2]
    return train_data, test_data
