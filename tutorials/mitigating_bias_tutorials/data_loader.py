import numpy as np
import pandas as pd


class DataLoader(object):
    def __init__(self, source, fair_column, fair_values, distance_columns):
        """
        Args:
                source (str) : name of the data source
                fair_column (str) : name of the column along which fairness needs to be enforced
                fair_values (list) : contains values that are acceptable in the 'fair_column'
                distance_columns (list) : contains values for computing distances between the points
        """
        assert len(fair_values) == 2, "Binary fair column supported currently."
        assert len(distance_columns) > 1, "Needs atleast one distance column."

        self.source = source
        self.fair_column = fair_column
        self.fair_values = fair_values
        self.distance_columns = distance_columns

    def load(self, normalize=False):
        """
        Loads and encodes the data.

        Args:
                normalize (bool) : Indicates whether the input data needs to be normalized
        """
        if self.source == "bank":
            self.data = pd.read_csv("data/bank-full.csv", sep=";")

        elif self.source == "census":
            self.data = pd.read_csv("data/uci_census.csv")

        elif self.source == "diabetes":
            self.data = pd.read_csv("data/diabetic_data.csv")
            age_buckets = {
                "[70-80)": 75,
                "[60-70)": 65,
                "[50-60)": 55,
                "[80-90)": 85,
                "[40-50)": 45,
                "[30-40)": 35,
                "[90-100)": 95,
                "[20-30)": 25,
                "[10-20)": 15,
                "[0-10)": 5,
            }
            self.data["age"] = self.data.apply(lambda x: age_buckets[x["age"]], axis=1)
            print("Bucketizing age.")

        else:
            raise ValueError(
                "Please specify a valid value for data source. %s is invalid."
                % (self.source)
            )

        self.data = self.data[self.data[self.fair_column].isin(self.fair_values)]

        fair_counts = sorted(
            dict(
                zip(
                    self.data[self.fair_column].value_counts().index,
                    self.data[self.fair_column].value_counts().values,
                )
            ).items(),
            key=lambda x: x[1],
            reverse=True,
        )
        assert len(fair_counts) == 2, "Fair counts should equal two."
        print("\nDistribution of '%s' column - " % (self.fair_column), fair_counts)

        print(
            "\nUsing these features in addition to %s for clustering - "
            % (self.fair_column),
            self.distance_columns,
        )
        self.data = self.data[[self.fair_column] + self.distance_columns].copy()

        self.data[self.fair_column] = np.where(
            self.data[self.fair_column] == fair_counts[0][0], 1, 0
        )
        print(
            "\nEncoding %s as 1, and %s as 0." % (fair_counts[0][0], fair_counts[1][0])
        )

        if normalize:
            print("\nNormalizing the data.")
            for col in self.data.columns:
                col_min, col_max = np.min(self.data[col]), np.max(self.data[col])
                self.data[col] = (self.data[col] - col_min) / (col_max - col_min)

    def split(self, split_size, random_state):
        """
        Splits the data based on fair column.

        Args:
                split_size (tuple) : Contains the split values for (majority, minority) class
                random_state (int) : Random state

        Returns:
                blues (list) : Indexes for the majority class
                reds (list) : Indexes for the minority class
        """
        df = (
            self.data[self.data[self.fair_column] == 1]
            .sample(max(split_size), random_state=random_state)
            .append(
                self.data[self.data[self.fair_column] == 0].sample(
                    min(split_size), random_state=random_state
                ),
                ignore_index=True,
            )
        )
        df = df.sample(frac=1, random_state=random_state)
        self.data = df.reset_index(drop=True)

        blues = list(self.data[self.data[self.fair_column] == 1].index)
        reds = list(self.data[self.data[self.fair_column] == 0].index)

        self.data_list = [list(i) for i in np.array(self.data)]

        return blues, reds
