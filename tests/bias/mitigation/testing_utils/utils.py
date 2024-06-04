import pandas as pd
import pytest
from sklearn import metrics


from holisticai.datasets import load_dataset, concatenate_datasets
# dictionnary of metrics
metrics_dict = {
    "Accuracy": metrics.accuracy_score,
    "Balanced accuracy": metrics.balanced_accuracy_score,
    "Precision": metrics.precision_score,
    "Recall": metrics.recall_score,
    "F1-Score": metrics.f1_score,
}

SHARD_SIZE=50

@pytest.fixture
def categorical_dataset():
    dataset = load_dataset("adult") # x,y,p_attr
    dataset = dataset.rename({"x":"X"}) # X,y,p_attr
    dataset = dataset.map(lambda x: {'group_a': x['p_attr']['group_a'], 'group_b': x['p_attr']['group_b']}) # X, y, p_attr, group_a, group_b
    dataset = dataset.groupby(['y','group_a']).head(SHARD_SIZE) # 0-ga | 0-gb  | 1-ga | 1-gb
    return dataset.train_test_split(test_size=0.2, stratify=dataset['y'], random_state=0)

@pytest.fixture
def regression_dataset():
    dataset = load_dataset("us_crime")
    dataset = dataset.rename({"x":"X"})
    dataset = dataset.select(range(2*SHARD_SIZE))
    dataset = dataset.map(lambda x: {'group_a': x['p_attr']['group_a'], 'group_b': x['p_attr']['group_b']})
    return dataset.train_test_split(test_size=0.2, random_state=0)

@pytest.fixture
def multiclass_dataset():
    dataset = load_dataset("us_crime_multiclass")
    dataset = dataset.rename({"x":"X"})
    dataset = dataset.map(lambda x: {'group_a': x['p_attr']['group_a'], 'group_b': x['p_attr']['group_b']})
    dataset = dataset.groupby(['y','group_a']).head(SHARD_SIZE)
    return dataset.train_test_split(test_size=0.2, stratify=dataset['y'], random_state=0)


@pytest.fixture
def small_clustering_dataset():
    dataset = load_dataset("clinical_records")
    dataset = dataset.map(lambda x: {'group_a': x['p_attr']['group_a'], 'group_b': x['p_attr']['group_b']})
    return dataset.train_test_split(test_size=0.2, stratify=dataset['y'], random_state=0)


@pytest.fixture
def small_recommender_dataset():
    return load_test_dataset("recommender", "small")


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


# efficacy metrics dataframe helper tool
def metrics_dataframe(y_pred, y_true, metrics_dict=metrics_dict):
    """
    Creates a dataframe containing the efficacy metrics

    Parameters
    ----------
    y_pred : array, shape=(n_samples,)
        The predicted values
    y_true : array, shape=(n_samples,)
        The true values
    metrics_dict : dict
        The dictionary of metrics to use

    Returns
    -------
    pandas.DataFrame
        The dataframe containing the efficacy metrics
    """
    metric_list = [[pf, fn(y_true, y_pred)] for pf, fn in metrics_dict.items()]
    return pd.DataFrame(metric_list, columns=["Metric", "Value"]).set_index("Metric")


def format_result_colum(name, config):
    """
    Formats a result column

    Parameters
    ----------
    name : str
        The name of the column
    config : dict
        The configuration

    Returns
    -------
    pandas.DataFrame
        The formatted column
    """
    return config["result"].rename(columns={"Value": name}).iloc[:, 0]


def show_result_table(configurations, df_baseline):
    """
    Shows the result table

    Parameters
    ----------
    configurations : dict
        The configurations
    df_baseline : pandas.DataFrame
        The baseline dataframe

    Returns
    -------
    pandas.DataFrame
        The result table
    """
    table = pd.concat(
        [df_baseline.iloc[:, 0]]
        + [format_result_colum(name, config) for name, config in configurations.items()]
        + [df_baseline.iloc[:, 1]],
        axis=1,
    )
    return table.rename(columns={"Value": "Baseline"})


def check_results(df1, df2):
    """
    Checks if the results are equal

    Parameters
    ----------
    df1 : pandas.DataFrame
        The first dataframe
    df2 : pandas.DataFrame
        The second dataframe

    Returns
    -------
    pandas.DataFrame
        The dataframe containing the results
    """
    import numpy as np
    assert np.isclose(df1["Value"].iloc[0], df2["Value"].iloc[0]), (df1["Value"].iloc[0], df2["Value"].iloc[0])


def fit(model, small_categorical_dataset):
    """
    Fits a model

    Parameters
    ----------
    model : object
        The model to fit
    small_categorical_dataset : tuple
        The dataset to use for fitting

    Returns
    -------
    object
        The fitted model
    """
    train_data, _ = small_categorical_dataset
    X, y, group_a, group_b = train_data

    fit_params = {"bm__group_a": group_a, "bm__group_b": group_b}
    model.fit(X, y, **fit_params)
    return model


def evaluate_pipeline(pipeline, small_categorical_dataset, metric_names, thresholds):
    """
    Evaluates a pipeline

    Parameters
    ----------
    pipeline : object
        The pipeline to evaluate
    small_categorical_dataset : tuple
        The dataset to use for evaluation
    metric_names : list
        The list of metric names to use
    thresholds : list

    Returns
    -------
    object
        The fitted pipeline
    """
    from holisticai.metrics import bias as metrics

    train_data, _ = small_categorical_dataset
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
