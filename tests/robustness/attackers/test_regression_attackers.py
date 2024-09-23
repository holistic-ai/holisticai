import warnings
warnings.filterwarnings("ignore")

from holisticai.datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
from holisticai.robustness.attackers import LinRegGDPoisoner
import pytest

SHARD_SIZE=60
POISON_PROPORTION=0.2

@pytest.fixture
def regression_dataset():
    dataset = load_dataset('us_crime', preprocessed=True)
    dataset = dataset.sample(SHARD_SIZE, random_state=42)
    train_test = dataset.train_test_split(test_size=0.75, random_state=42)
    train = train_test['train']
    test = train_test['test']
    return train, test

def test_gdblinear_complete(regression_dataset):
    train, test = regression_dataset
    X_train = train['X'].drop(columns=['fold'])
    X_test = test['X'].drop(columns=['fold'])

    y_train = train['y']
    y_test = test['y']

    # Standardize data and fit model
    pipe = Pipeline([('lr', linear_model.LinearRegression())])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    baseline_error = mean_squared_error(y_test, y_pred)

    categorical_mask = np.zeros(X_train.shape[1])
    categorical_mask[0] = 0

    poiser = LinRegGDPoisoner(poison_proportion=0.2, num_inits=1, max_iter=2) #  

    # Poison the training data
    x_poised, y_poised = poiser.generate(X_train, y_train, categorical_mask = categorical_mask, return_only_poisoned=False)

    pipe_poisoned = Pipeline([('lr', linear_model.LinearRegression())])
    pipe_poisoned.fit(x_poised, y_poised)

    y_adv_pred = pipe_poisoned.predict(X_test)

    poised_err = mean_squared_error(y_test, y_adv_pred)

    poisoned_samples = int(X_train.shape[0] * POISON_PROPORTION / (1 - POISON_PROPORTION) + 0.5)

    assert x_poised.shape[0] == X_train.shape[0] + poisoned_samples
    assert poised_err != baseline_error


def test_gdblinear_only_poison(regression_dataset):
    train, test = regression_dataset
    X_train = train['X'].drop(columns=['fold'])
    X_test = test['X'].drop(columns=['fold'])

    y_train = train['y']
    y_test = test['y']

    # Standardize data and fit model
    pipe = Pipeline([('lr', linear_model.LinearRegression())])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    baseline_error = mean_squared_error(y_test, y_pred)

    categorical_mask = np.zeros(X_train.shape[1])
    categorical_mask[0] = 0

    poiser = LinRegGDPoisoner(poison_proportion=0.2, num_inits=1, max_iter=2) #  

    # Poison the training data
    x_poised, y_poised = poiser.generate(X_train, y_train, categorical_mask = categorical_mask, return_only_poisoned=True)

    pipe_poisoned = Pipeline([('lr', linear_model.LinearRegression())])
    pipe_poisoned.fit(x_poised, y_poised)

    y_adv_pred = pipe_poisoned.predict(X_test)

    poised_err = mean_squared_error(y_test, y_adv_pred)

    poisoned_samples = int(X_train.shape[0] * POISON_PROPORTION / (1 - POISON_PROPORTION) + 0.5)

    assert x_poised.shape[0] == poisoned_samples
    assert y_poised.shape[0] == poisoned_samples
    assert poised_err != baseline_error
