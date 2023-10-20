import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from holisticai.datasets import load_adult
from holisticai.utils.trade_off_analysers import Fairea
from tests.bias.mitigation.testing_utils.utils import small_categorical_dataset


def test_fairea_pipeline(small_categorical_dataset):
    """
    Test the Fairea class and its methods.
    """
    # Create an instance of the Fairea class
    fairea = Fairea()

    # Generate some example input data using the small_categorical_dataset function
    train_data, test_data = small_categorical_dataset
    x_train, y_train, group_a_train, group_b_train = train_data
    x_test, y_test, group_a_test, group_b_test = test_data

    # Create the baseline model
    fairea.create_baseline(x_train, y_train, group_a_train, group_b_train)

    # Check that the baseline accuracy and fairness were calculated correctly
    assert (fairea.baseline_acc <= 1).all()

    # Check that the MinMaxScaler objects were created and fit correctly
    assert isinstance(fairea.acc_scaler, MinMaxScaler)
    assert isinstance(fairea.fair_scaler, MinMaxScaler)
    assert fairea.acc_norm.max() <= 1.0
    assert fairea.acc_norm.min() >= 0.0
    assert fairea.fairness_norm.max() <= 1.0
    assert fairea.fairness_norm.min() >= 0.0

    # Generate some example predictions for a new model
    y_pred = np.random.randint(2, size=y_test.shape)

    # Add the outcomes of the new model
    fairea.add_model_outcomes("new_model", y_test, y_pred, group_a_test, group_b_test)

    # Check that the outcomes were added correctly
    assert isinstance(fairea.methods["new_model"], tuple)
    assert isinstance(fairea.methods["new_model"], tuple)

    # Check that the MinMaxScaler objects were updated correctly
    assert isinstance(fairea.acc_scaler, MinMaxScaler)
    assert isinstance(fairea.fair_scaler, MinMaxScaler)

    # Check if the plot methods run without errors
    ax = fairea.plot_baseline()
    assert isinstance(ax, plt.Axes)

    ax = fairea.plot_methods()
    assert isinstance(ax, plt.Axes)

    # Check if the methods are returning the expected value types
    assert isinstance(fairea.get_best(), (str, type(None)))

    assert isinstance(fairea.region_classification(), pd.DataFrame)

    assert isinstance(fairea.determine_area(), (pd.DataFrame, type(None)))
