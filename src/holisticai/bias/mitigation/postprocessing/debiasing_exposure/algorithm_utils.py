import math

import numpy as np
import pandas as pd


def filter_invalid_examples(rankings, query_col, group_col):
    new_rankings = []
    for _, ranking in rankings.groupby(query_col):
        if (ranking[group_col].sum() > 0).any():
            new_rankings.append(ranking)
    new_rankings = pd.concat(new_rankings, axis=0).reset_index(drop=True)
    return new_rankings


def exposure_metric(rankings, group_col: str, query_col: str, score_col: str):
    rankings = filter_invalid_examples(rankings, query_col=query_col, group_col=group_col)

    exp_diff = np.mean(
        [exposure_diff(ranking[score_col], ranking[group_col]) for q, ranking in rankings.groupby(query_col)]
    )

    metric_values = [
        exposure_ratio(ranking[score_col], ranking[group_col]) for q, ranking in rankings.groupby(query_col)
    ]
    exp_ratio = np.mean([m for m in metric_values if not math.isnan(m)])

    df = pd.DataFrame([{"exposure_ratio": exp_ratio, "exposure difference": exp_diff}]).T
    df.columns = ["Value"]
    return df


def exposure_diff(data_per_query, prot_idx_per_query):
    """
    Description
    -----------
    Computes the exposure difference between protected and non-protected groups.

    Parameters
    ----------
    data: matrix-like
        all predictions.

    prot_idx: matrix-like
        list states which item is protected or non-protected.

    Return
    ------
        float value
    """
    (
        judgments_per_query,
        protected_items_per_query,
        nonprotected_items_per_query,
    ) = find_items_per_group_per_query(data_per_query, prot_idx_per_query)

    exposure_prot = normalized_exposure(protected_items_per_query, judgments_per_query)
    exposure_nprot = normalized_exposure(nonprotected_items_per_query, judgments_per_query)
    exposure_diff = np.maximum(0, (exposure_nprot - exposure_prot))

    return exposure_diff


def exposure_ratio(data_per_query, prot_idx_per_query):
    """
    Description
    -----------
    Computes the exposure difference between protected and non-protected groups.

    Parameters
    ----------
    data: matrix-like
        all predictions.

    prot_idx: matrix-like
        list states which item is protected or non-protected.

    Return
    ------
        float value
    """
    (
        judgments_per_query,
        protected_items_per_query,
        nonprotected_items_per_query,
    ) = find_items_per_group_per_query(data_per_query, prot_idx_per_query)

    exposure_prot = normalized_exposure(protected_items_per_query, judgments_per_query)
    exposure_nprot = normalized_exposure(nonprotected_items_per_query, judgments_per_query)
    exposure_diff = exposure_nprot / exposure_prot

    return exposure_diff


def find_items_per_group_per_query(data, protected_feature):
    data_per_query = np.array(data).astype(np.float32)
    if np.any(np.isnan(data_per_query)):
        raise ValueError("data has NaN values!!, fix your data or normalize it before training.")
    if len(data_per_query.shape) == 1:
        data_per_query = data_per_query[:, None]
    protected_feature = np.array(protected_feature)
    protected_items_per_query = data_per_query[protected_feature]
    nonprotected_items_per_query = data_per_query[~protected_feature]
    return data_per_query, protected_items_per_query, nonprotected_items_per_query


def normalized_exposure(group_data, all_data):
    """
    Description
    -----------
    Calculates the exposure of a group in the entire ranking.

    Parameters
    ----------
    group_data: matrix-like
        predictions of relevance scores for one group.

    all_data: matrix-like
        all predictions.

    Return
    ------
    float value that is normalized exposure in a ranking for one group.
    """
    return (np.sum(topp_prot(group_data, all_data) / np.log(2))) / group_data.size


def normalized_topp_prot_deriv_per_group(group_features, all_features, group_predictions, all_predictions):
    """
    Description
    -----------
    Normalizes the results of the derivative of topp_prot.

    Parameters
    ----------
    group_features: array-like
        feature vector of (non-) protected group.

    all_features: array-like
        feature vectors of all data points.

    group_predictions: array-like
        predictions of all data points.

    all_predictions: array-like
        predictions of all data points.

    Return
    numpy array of float values.
    """
    derivative = topp_prot_first_derivative(group_features, all_features, group_predictions, all_predictions)
    result = (np.sum(derivative / np.log(2), axis=0)) / group_predictions.size
    return result


def topp_prot(group_items, all_items):
    """
    Description
    -----------
    Given a dataset of features what is the probability of being at the top position.

    Parameters
    ----------
    group_items:
        Vector of predicted scores of one group (protected or non-protected).

    all_items:
        Vector of predicted scores of all items.

    Return
    ------
        Numpy array of float values.
    """
    return np.exp(group_items) / np.sum(np.exp(all_items))


def topp_prot_first_derivative(group_features, all_features, group_predictions, all_predictions):
    """
    Description
    -----------
    Derivative for topp_prot in pieces.

    Parameters
    ----------
    group_features: array-like
        Feature vector of (non-) protected group.

    group_predictions: array-like
        Predicted scores for (non-) protected group.

    all_predictions: array-like
        Predictions of all data points.

    all_features: array-like
        Feature vectors of all data points.

    Return
    ------
    numpy array with weight adjustments.
    """

    numerator1 = np.dot(np.transpose(np.exp(group_predictions)), group_features)
    numerator2 = np.sum(np.exp(all_predictions))
    numerator3 = np.sum(np.dot(np.transpose(np.exp(all_predictions)), all_features))
    denominator = np.sum(np.exp(all_predictions)) ** 2

    result = (numerator1 * numerator2 - np.exp(group_predictions) * numerator3) / denominator

    # return result as flat numpy array instead of matrix
    return np.asarray(result)


def topp(v):
    """
    Description
    -----------
    Computes the probability of a document being.

    Parameters
    ----------
    v: array-like
        all training judgments or all predictions.

    Return:
    float value which is a probability.

    """
    return np.exp(v) / np.sum(np.exp(v))


def hh(q, x):
    return (q, hash(str(x)))


class Standarizer:
    def __init__(self, group_col):
        self.group_col = group_col

    def fit_transform(self, feature_matrix):
        self._mus = np.array(feature_matrix).mean()
        self._sigmas = np.array(feature_matrix).std()
        protected_feature = feature_matrix[self.group_col]
        feature_matrix = (feature_matrix - self._mus) / self._sigmas
        feature_matrix[self.group_col] = protected_feature
        return feature_matrix

    def transform(self, feature_matrix):
        protected_feature = feature_matrix[self.group_col]
        feature_matrix = (feature_matrix - self._mus) / self._sigmas
        feature_matrix[self.group_col] = protected_feature
        return feature_matrix
