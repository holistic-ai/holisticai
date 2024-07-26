import numpy as np
import pandas as pd
from holisticai.utils._recommender_tools import entropy
from holisticai.utils._validation import _clustering_checks

# sklearn imports
from sklearn.metrics import (
    adjusted_mutual_info_score,
    mean_absolute_error,
    silhouette_samples,
)


def cluster_balance(group_a, group_b, y_pred):
    """Cluster Balance

    Given a clustering and protected attribute. The cluster balance is\
    the minimum over all groups and clusters of the ratio of the representation\
    of members of that group in that cluster to the representation overall.

    Interpretation
    --------------
    A value of 1 is desired. That is when all clusters have the exact same\
    representation as the data. Lower values imply the existence of clusters\
    where either group_a or group_b is underrepresented.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Cluster predictions (categorical)

    Returns
    -------
    float
        Cluster Balance

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import cluster_balance
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred_cluster = np.array([0, 1, 1, 2, 0, 0, 0, 0, 1, 2])
    >>> cluster_balance(group_a, group_b, y_pred_cluster)
    0.5
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, _, _ = _clustering_checks(group_a, group_b, y_pred)

    # Get clusters
    clusters = np.unique(y_pred)

    # group_a ratio overall
    r_tot_a = group_a.sum() / len(group_a)
    # group_b ratio overall
    r_tot_b = group_b.sum() / len(group_b)

    min_ratio = 1

    # loop over clusters
    for c in clusters:
        # variables
        members = y_pred == c
        n_members = members.sum() + 1.0e-20
        n_a = group_a[members].sum() + 1.0e-20
        n_b = group_b[members].sum() + 1.0e-20

        # group_a ratios
        ratio_a = (n_a / n_members) / r_tot_a
        min_a = min(ratio_a, 1 / ratio_a)

        # group_b ratios
        ratio_b = (n_b / n_members) / r_tot_b
        min_b = min(ratio_b, 1 / ratio_b)

        min_ratio = min(min_ratio, min_a, min_b)

    # return minimum balance in list
    return min_ratio


def min_cluster_ratio(group_a, group_b, y_pred):
    """Minimum Cluster Ratio

    Given a clustering and protected attributes. The min cluster ratio is\
    the minimum over all clusters of the ratio of number of group_a members\
    to the number of group_b members.

    Interpretation
    --------------
    A value of 1 is desired. That is when all clusters are perfectly\
    balanced. Low values imply the existence of clusters where\
    group_a has fewer members than group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Cluster predictions (categorical)

    Returns
    -------
    float
        Minimum Cluster Ratio

    Examples
    --------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import min_cluster_ratio
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred_cluster = np.array([0, 1, 1, 2, 0, 0, 0, 0, 1, 2])
    >>> min_cluster_ratio(group_a, group_b, y_pred_cluster)
    0.2499999999375
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, _, _ = _clustering_checks(group_a, group_b, y_pred)

    # Get clusters
    clusters = np.unique(y_pred)
    min_ratio = np.inf

    # Get balance of each cluster
    for c in clusters:
        members = y_pred == c
        n_a = group_a[members].sum()
        n_b = group_b[members].sum()

        min_ratio = min(min_ratio, (n_a / (n_b + 1.0e-32)))

    return min_ratio


def _avg_cluster_ratio(group_a, group_b, y_pred):
    """Average Cluster Ratio

    Given a clustering and protected attributes. The average cluster ratio is\
    the average over all clusters of the ratio of group_a members to group_b\
    members in that cluster.

    Interpretation
    --------------
    A value of 1 is desired. Low values imply the predominance of clusters where\
    group_a is underrepresented. High values imply the predominance of clusters where\
    group_b is underrepresented.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Cluster predictions (categorical)

    Returns
    -------
    float
        Average Cluster Ratio

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import avg_cluster_ratio
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred_cluster = np.array([0, 1, 1, 2, 0, 0, 0, 0, 1, 2])
    >>> avg_cluster_ratio(group_a, group_b, y_pred_cluster)
    1.0833333323124998
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, _, _ = _clustering_checks(group_a, group_b, y_pred)

    # Get clusters
    clusters = np.unique(y_pred)
    n_clusters = len(clusters)
    balances = np.zeros((n_clusters,))

    # Get balance of each cluster
    for i, c in zip(range(n_clusters), clusters):
        members = y_pred == c
        n_a = group_a[members].sum()
        n_b = group_b[members].sum()
        # if n_b zero we get infinity for mean
        if n_b == 0:
            return np.inf
        balances[i] = n_a / n_b
        # Check non zero for b
        balances[i] = n_a / (n_b + 1.0e-32)

    return balances.mean()


def _cluster_dist(y_pred_g, clusters):
    """Group distribution over clusters

    This function computes the distribution of the group across clusters.

    Parameters
    ----------
    y_pred_g : array-like
        Cluster predictions (categorical)
    clusters : array-like
        Cluster ground truth (categorical)

    Returns
    -------
    numpy array
        Cluster Distribution
    """
    bin_mat = y_pred_g.reshape(-1, 1) == clusters.reshape(1, -1)
    dist = bin_mat.sum(axis=0)
    return dist / dist.sum()


def cluster_dist_l1(group_a, group_b, y_pred):
    """Cluster Distribution Total Variation

    This function computes the distribution of group_a and group_b across clusters.\
    It then outputs the total variation distance between these distributions.

    Interpretation
    --------------
    A value of 0 is desired. That indicates that both groups are distributed\
    similarly amongst the clusters. The metric ranges between 0 and 1,\
    with higher values indicating the groups are distributed in very\
    different ways.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Cluster predictions (categorical)

    Returns
    -------
    float
        Cluster Distribution Total Variation

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import cluster_dist_l1
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred_cluster = np.array([0, 1, 1, 2, 0, 0, 0, 0, 1, 2])
    >>> cluster_dist_l1(group_a, group_b, y_pred_cluster)
    0.4166666666666667
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, _, _ = _clustering_checks(group_a, group_b, y_pred)

    # get unique clusters
    clusters = np.unique(y_pred)

    # split data by group
    y_pred_a = y_pred[group_a == 1]
    y_pred_b = y_pred[group_b == 1]

    # compute distributions
    dist_a = _cluster_dist(y_pred_a, clusters)
    dist_b = _cluster_dist(y_pred_b, clusters)

    # return total variation norm
    return 0.5 * len(dist_a) * mean_absolute_error(dist_a, dist_b)


def cluster_dist_kl(group_a, group_b, y_pred):
    """Cluster Distribution KL

    This function computes the distribution of group_a and group_b\
    membership across the clusters. It then returns the KL distance\
    from the distribution of group_a to the distribution of group_b.

    Interpretation
    --------------
    A value of 0 is desired. That indicates that both groups are distributed\
    similarly amongst the clusters. Higher values indicate the distributions\
    of both groups amongst the clusters differ more.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Cluster predictions (categorical)

    Returns
    -------
    float
        Cluster Distribution KL

    Notes
    -----
    :math:`KL(P_a,P_b)`

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import cluster_dist_kl
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred_cluster = np.array([0, 1, 1, 2, 0, 0, 0, 0, 1, 2])
    >>> cluster_dist_kl(group_a, group_b, y_pred_cluster)
    0.4054651081081642
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, _, _ = _clustering_checks(group_a, group_b, y_pred)

    # get unique clusters
    clusters = np.unique(y_pred)

    # split data by group
    y_pred_a = y_pred[group_a == 1]
    y_pred_b = y_pred[group_b == 1]

    # compute distributions
    dist_a = _cluster_dist(y_pred_a, clusters)
    dist_b = _cluster_dist(y_pred_b, clusters)

    # return KL
    return entropy(dist_a, dist_b)


def cluster_dist_entropy(group, y_pred):
    """Minority Cluster Distribution Entropy

    The entropy of the distribution of the group
    over the clusters.

    Interpretation
    --------------
    Lower values indicate most members of the group are allocated to\
    the same cluaster. Hence we encourage higher values of\
    the entropy, which indicate more homogeneity.

    Parameters
    ----------
    group : array-like
        Group membership vector (binary)
    y_pred : array-like
        Cluster predictions (categorical)

    Returns
    -------
    float
        Group Presence Entropy

    Notes
    -----
    :math:`Entropy(P_{group})`

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import cluster_dist_entropy
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred_cluster = np.array([0, 1, 1, 2, 0, 0, 0, 0, 1, 2])
    >>> cluster_dist_entropy(group_b, y_pred_cluster)
    0.8675632284814613
    """
    # check and coerce inputs
    group, _, y_pred, _, _, _ = _clustering_checks(group, group, y_pred)

    # get unique clusters
    clusters = np.unique(y_pred)

    # split data by group
    y_pred_group = y_pred[group == 1]

    # compute distribution
    dist_b = _cluster_dist(y_pred_group, clusters)

    # return Entropy
    return entropy(dist_b)


def _ami_diff(group_a, group_b, y_pred, y_true):
    """Adjusted Mutual information Difference

    We compute the difference of the adjusted mutual information\
    on group_a and group_b.

    Interpretation
    --------------
    The MI difference ranges from -1 to 1, with lower values indicating bias\
    towards group_a and larger values indicating bias against group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Cluster predictions (categorical)
    y_true : array-like
        Cluster ground truth (categorical)

    Returns
    -------
    float
        Mutual information Difference

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import ami_diff
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0, 1, 1, 2, 0, 0, 0, 0, 1, 2])
    >>> y_true = np.array([0, 1, 1, 2, 1, 0, 1, 2, 0, 2])
    >>> ami_diff(group_a, group_b, y_pred, y_true)
    0.6342556627317533
    """
    # check and coerce inputs
    group_a, group_b, y_pred, y_true, _, _ = _clustering_checks(group_a, group_b, y_pred, y_true)

    # Slice by min and maj groups
    y_pred_a = y_pred[group_a == 1]
    y_pred_b = y_pred[group_b == 1]
    y_true_a = y_true[group_a == 1]
    y_true_b = y_true[group_b == 1]

    # Compute AMI scores
    ami_a = adjusted_mutual_info_score(y_true_a, y_pred_a)
    ami_b = adjusted_mutual_info_score(y_true_b, y_pred_b)

    # Return Spread
    return ami_a - ami_b


def social_fairness_ratio(group_a, group_b, data, centroids):
    """Social Fairness Ratio

    Given a centroid based clustering, this function compute the average\
    distance to the nearest centroid for both groups. The metric is the\
    ratio of the resulting distance for group_a to group_b.

    Interpretation
    --------------
    A value of 1 is desired. Lower values indicate the group_a\
    is on average closer to the respective centroids. Higher\
    values indicate that group_a is on average further from the\
    respective centroids.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    data : matrix-like
        Data matrix of shape (num_inst, dim)
    centroids : matrix-like
        Centroids (centers) of shape (num_centroids, dim)

    Returns
    -------
    float
        Social Fairness Ratio

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import social_fairness_ratio
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> data = np.array(
    ...     [
    ...         [-1, 1],
    ...         [1, 1],
    ...         [1, 1],
    ...         [0, -1],
    ...         [-1, 1],
    ...         [-1, 1],
    ...         [-1, 1],
    ...         [-1, 1],
    ...         [1, 1],
    ...         [0, -1],
    ...     ]
    ... )
    >>> centroids = np.array([[-2, 1], [1, 2], [0, -2]])
    >>> social_fairness_ratio(group_a, group_b, data, centroids)
    1.0
    """
    # check and coerce inputs
    group_a, group_b, _, _, data, centroids = _clustering_checks(
        group_a, group_b, y_pred=None, y_true=None, data=data, centroids=centroids
    )

    # Split by group
    data_a = data[group_a == 1]
    data_b = data[group_b == 1]

    # Reshape matrices for vectorization
    data_a = data_a.reshape(data_a.shape[0], 1, data_a.shape[1])
    data_b = data_b.reshape(data_b.shape[0], 1, data_b.shape[1])
    centroids = centroids.reshape(1, centroids.shape[0], centroids.shape[1])

    # Calculate distances
    dist_a = np.sqrt(((data_a - centroids) ** 2).sum(axis=-1))
    dist_b = np.sqrt(((data_b - centroids) ** 2).sum(axis=-1))

    # Take minimum over centroids and average over instances
    dist_a = dist_a.min(axis=1).mean(axis=0)
    dist_b = dist_b.min(axis=1).mean(axis=0)

    # return ratio of averages
    return dist_a / dist_b


def silhouette_diff(group_a, group_b, data, y_pred):
    """Silhouette Difference

    We compute the difference of the mean silhouette score for both\
    groups.

    Interpretation
    --------------
    The silhouette difference ranges from -1 to 1, with lower values indicating bias\
    towards group_a and larger values indicating bias against group_b.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    data : matrix-like
        Data matrix of shape (num_inst, dim)
    y_pred : array-like
        Cluster predictions (categorical)

    Returns
    -------
    float
        Silhouette difference

    Notes
    -----
    :math:`\texttt{mean_silhouette_a - mean_silhouette_b}`

    Examples
    -------
    >>> import numpy as np
    >>> from holisticai.bias.metrics import silhouette_diff
    >>> group_a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    >>> group_b = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    >>> data = np.array(
    ...     [
    ...         [-1, 1],
    ...         [1, 1],
    ...         [1, 1],
    ...         [0, -1],
    ...         [-1, 1],
    ...         [-1, 1],
    ...         [-1, 1],
    ...         [-1, 1],
    ...         [1, 1],
    ...         [0, -1],
    ...     ]
    ... )
    >>> y_pred = np.array([0, 1, 1, 2, 0, 0, 0, 0, 1, 2])
    >>> silhouette_diff(group_a, group_b, data, y_pred)
    0.0
    """
    # check and coerce inputs
    group_a, group_b, y_pred, _, data, _ = _clustering_checks(group_a, group_b, y_pred, data=data)

    # Compute silhouette scores
    scores = silhouette_samples(data, y_pred, metric="euclidean")

    # Split min and maj
    scores_a = scores[group_a == 1]
    scores_b = scores[group_b == 1]

    return 0.5 * (np.mean(scores_a) - np.mean(scores_b))


def clustering_bias_metrics(
    group_a,
    group_b,
    y_pred,
    data=None,
    centroids=None,
    metric_type="equal_outcome",
):
    """Clustering bias metrics batch computation

    This function computes all the relevant clustering bias metrics,\
    and displays them as a pandas dataframe.

    Parameters
    ----------
    group_a : array-like
        Group membership vector (binary)
    group_b : array-like
        Group membership vector (binary)
    y_pred : array-like
        Cluster predictions (categorical)
    data : matrix-like, optional
        Data matrix of shape (num_inst, dim)
    centroids : matrix-like, optional
        Centroids (centers)
    metric_type : str, optional
        Specifies which metrics we compute: 'both', 'equal_outcome' or 'equal_opportunity'

    Returns
    -------
    pandas DataFrame
        Metrics | Values | Reference
    """
    perform = {
        "Cluster Balance": cluster_balance,
        "Minimum Cluster Ratio": min_cluster_ratio,
        "Cluster Distribution Total Variation": cluster_dist_l1,
        "Cluster Distribution KL Div": cluster_dist_kl,
    }

    centroids_data_perform = {"Social Fairness Ratio": social_fairness_ratio}

    pred_data_perform = {"Silhouette Difference": silhouette_diff}

    ref_vals = {
        "Cluster Balance": 1,
        "Minimum Cluster Ratio": 1,
        "Cluster Distribution Total Variation": 0,
        "Cluster Distribution KL Div": 0,
        "Social Fairness Ratio": 1,
        "Silhouette Difference": 0,
        "Mutual Information Difference": 0,
    }

    out_metrics = []
    opp_metrics = []

    out_metrics += [[pf, fn(group_a, group_b, y_pred), ref_vals[pf]] for pf, fn in perform.items()]
    if data is not None and centroids is not None:
        out_metrics += [
            [pf, fn(group_a, group_b, data, centroids), ref_vals[pf]] for pf, fn in centroids_data_perform.items()
        ]
    if data is not None and y_pred is not None:
        out_metrics += [[pf, fn(group_a, group_b, data, y_pred), ref_vals[pf]] for pf, fn in pred_data_perform.items()]

    if metric_type == "both":
        metrics = out_metrics + opp_metrics
        return pd.DataFrame(metrics, columns=["Metric", "Value", "Reference"]).set_index("Metric")

    if metric_type == "equal_outcome":
        return pd.DataFrame(out_metrics, columns=["Metric", "Value", "Reference"]).set_index("Metric")

    if metric_type == "equal_opportunity":
        return pd.DataFrame(opp_metrics, columns=["Metric", "Value", "Reference"]).set_index("Metric")

    msg = "metric_type is not one of : both, equal_outcome, equal_opportunity"
    raise ValueError(msg)
