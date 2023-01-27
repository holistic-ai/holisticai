import pandas as pd
from sklearn.model_selection import train_test_split


def preprocessed_dataset(dataset="adult", splitted=True):
    if dataset == "adult":
        from holisticai.datasets import load_adult

        # Dataset
        dataset = load_adult()

        # Dataframe
        df = pd.concat([dataset["data"], dataset["target"]], axis=1)
        protected_variables = ["sex", "race"]
        output_variable = ["class"]

        # Simple preprocessing
        y = df[output_variable].replace({">50K": 1, "<=50K": 0})
        X = pd.get_dummies(df.drop(protected_variables + output_variable, axis=1))
        group = ["sex"]
        group_a = df[group] == "Female"
        group_b = df[group] == "Male"
        data = [X.iloc[:, :-1], y, group_a, group_b]

        # Train test split
        dataset = train_test_split(*data, test_size=0.2, shuffle=True)
        if splitted:
            train_data = dataset[::2]
            test_data = dataset[1::2]
            return train_data, test_data
        else:
            return data
