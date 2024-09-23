import numpy as np
from holisticai.utils.surrogate_models import get_features, get_number_of_rules


class WeightedAverageDepth:
    """
    Represents the Weighted Average Depth metric.
    """

    reference: int = 0
    name: str = "Weighted Average Depth"

    def __init__(self, ignore_repeated: bool = True):
        self.ignore_repeated = ignore_repeated

    def __call__(self, tree):
        """
        Calculates the weighted average depth of a tree.

        Parameters
        ----------
        tree: Tree
            The tree to calculate the weighted average depth of.

        Returns:
            float: The weighted average depth value.
        """

        depths, counts = get_depths_counts(0, tree, [], [])
        n_samples = sum(counts)
        depths = np.array(depths)
        weights = np.array(counts) / n_samples
        return (depths * weights).sum()


def weighted_average_depth(tree):
    """
    Weighted Average Depth calculates the average depth of a tree considering the number
    of samples that pass through each cut.

    Parameters
    ----------
    tree: Tree
        The tree to calculate the weighted average depth of.

    Returns
    -------
    float
        The weighted average depth value

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from holisticai.explainability.metrics import weighted_average_depth
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = DecisionTreeClassifier()
    >>> clf.fit(X, y)
    >>> weighted_average_depth(clf.tree_)

    Reference
    ----------
        Laber, E., Murtinho, L., & Oliveira, F. (2023).
        Shallow decision trees for explainable k-means clustering.
        Pattern Recognition, 137, 109239.
    """
    metric = WeightedAverageDepth()
    return metric(tree)


class WeightedAverageExplainabilityScore:
    """
    Represents the Weighted Average Explainability Score.
    """

    reference: int = 0
    name: str = "Weighted Average Explainability Score"

    def __init__(self, ignore_repeated: bool = True):
        self.ignore_repeated = ignore_repeated

    def __call__(self, tree):
        """
        Calculates the weighted average depth of a tree.

        Parameters
        ----------
        tree: Tree
            The tree to calculate the weighted average depth of.

        Returns:
            float: The weighted average depth value.
        """

        depths, counts = get_cuts_counts(0, tree, [], [], set())
        n_samples = sum(counts)
        depths = np.array(depths)
        weights = np.array(counts) / n_samples
        return (depths * weights).sum()


def weighted_average_explainability_score(tree):
    """
    Weighted Average Explainability Score calculates the average depth of a tree considering the number
    of samples that pass through each cut.

    Parameters
    ----------
    tree: Tree
        The tree to calculate the weighted average depth of.

    Returns
    -------
    float
        The weighted average depth value

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from holisticai.explainability.metrics import (
    ...     weighted_average_explainability_score,
    ... )
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = DecisionTreeClassifier()
    >>> clf.fit(X, y)
    >>> weighted_average_explainability_score(clf.tree_)

    Reference
    ----------
        Laber, E., Murtinho, L., & Oliveira, F. (2023).
        Shallow decision trees for explainable k-means clustering.
        Pattern Recognition, 137, 109239.
    """
    metric = WeightedAverageExplainabilityScore()
    return metric(tree)


def is_leaf(node_index, tree):
    """
    Check if a node is a leaf.

    Parameters
    ----------
    node_index : int
        The index of the node to check.
    tree : Tree
        The tree to check the node in.

    Returns
    -------
    bool
        Whether the node is a leaf or not.
    """
    return tree.children_left[node_index] == -1 and tree.children_right[node_index] == -1


def get_cuts_counts(node_index, tree, cuts, counts, cur_set):
    """
    Get the cuts and counts of a tree.

    Parameters
    ----------
    node_index : int
        The index of the node to start from.
    tree : Tree
        The tree to get the cuts and counts from.
    cuts : list
        The list to store the cuts.
    counts : list
        The list to store the counts.
    cur_set : set
        The set to store the current cuts.

    Returns
    -------
    list
        The list of cuts.
    list
        The list of counts.
    """
    if is_leaf(node_index, tree):
        cuts.append(len(cur_set))
        counts.append(tree.n_node_samples[node_index])
    else:
        children_left_set = cur_set.copy()
        children_left_set.add((tree.feature[node_index], -1))
        children_right_set = cur_set.copy()
        children_right_set.add((tree.feature[node_index], 1))

        if tree.children_left[node_index] != -1:
            get_cuts_counts(tree.children_left[node_index], tree, cuts, counts, children_left_set)
        if tree.children_right[node_index] != -1:
            get_cuts_counts(tree.children_right[node_index], tree, cuts, counts, children_right_set)
    return cuts, counts


def get_depths_counts(node_index, tree, depths, counts, h=0):
    """
    Get the depths and counts of a tree.

    Parameters
    ----------
    node_index : int
        The index of the node to start from.
    tree : Tree
        The tree to get the depths and counts from.
    depths : list
        The list to store the depths.
    counts : list
        The list to store the counts.
    h : int, default=0
        The current depth.

    Returns
    -------
    list
        The list of depths.
    list
        The list of counts.
    """
    if is_leaf(node_index, tree):
        depths.append(h)
        counts.append(tree.n_node_samples[node_index])

    if tree.children_left[node_index] != -1:
        get_depths_counts(tree.children_left[node_index], tree, depths, counts, h + 1)
    if tree.children_right[node_index] != -1:
        get_depths_counts(tree.children_right[node_index], tree, depths, counts, h + 1)

    return depths, counts


class WeightedTreeGini:
    """
    Represents the Weighted Gini Index metric.
    """

    reference: float = 0.0
    name: str = "Weighted Gini Index"

    def __call__(self, tree):
        """
        Calculates the weighted Gini index of a tree.

        Parameters
        ----------
        tree: Tree
            The tree to calculate the weighted Gini index of.

        Returns:
            float: The weighted Gini index value.
        """

        def gini_impurity(node_index):
            node_samples = tree.n_node_samples[node_index]
            if node_samples == 0:
                return 0.0
            node_value = tree.value[node_index, 0, :]
            p = node_value / node_samples
            return 1.0 - np.sum(p**2)

        def variance_impurity(node_index):
            node_samples = tree.n_node_samples[node_index]
            if node_samples == 0:
                return 0.0
            node_value = tree.value[node_index, 0, :]
            mean_value = np.mean(node_value)
            return np.sum((node_value - mean_value) ** 2) / node_samples

        is_classification = tree.n_classes[0] > 1

        weighted_impurity = 0.0
        total_samples = tree.n_node_samples[0]

        def accumulate_impurity(node_index):
            nonlocal weighted_impurity
            if is_leaf(node_index, tree):
                node_samples = tree.n_node_samples[node_index]
                impurity = gini_impurity(node_index) if is_classification else variance_impurity(node_index)
                weighted_impurity += (node_samples / total_samples) * impurity
            else:
                accumulate_impurity(tree.children_left[node_index])
                accumulate_impurity(tree.children_right[node_index])

        accumulate_impurity(0)
        return weighted_impurity


def weighted_tree_gini(tree):
    """
    Compute the weighted Gini index for the tree (WGNI).
    Reference value: 0.0

    Parameters
    ----------
    tree : Tree
        The tree to compute the weighted Gini index of.

    Returns
    -------
    float
        The weighted Gini index of the tree.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from holisticai.explainability.metrics import weighted_average_depth
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = DecisionTreeClassifier()
    >>> clf.fit(X, y)
    >>> weighted_tree_gini(clf.tree_)
    """
    metric = WeightedTreeGini()
    return metric(tree)


class TreeDepthVariance:
    """
    Represents the Tree Depth Variance metric.
    """

    reference: float = 0.0
    name: str = "Tree Depth Variance"

    def __call__(self, tree):
        """
        Calculates the variance of the depths of the leaves in the tree.

        Parameters
        ----------
        tree: Tree
            The tree to calculate the depth variance of.

        Returns:
            float: The variance of the leaf depths.
        """
        depths, _ = get_depths_counts(0, tree, [], [])
        mean_depth = np.mean(depths)
        variance = np.mean((depths - mean_depth) ** 2)
        return variance


def tree_depth_variance(tree):
    """
    Compute the variance of the depths of the leaves in the tree (TDV).
    Reference value: 0.0

    Parameters
    ----------
    tree : Tree
        The tree to compute the depth variance of.

    Returns
    -------
    float
        The variance of the leaf depths.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from holisticai.explainability.metrics import weighted_average_depth
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = DecisionTreeClassifier()
    >>> clf.fit(X, y)
    >>> tree_depth_variance(clf.tree_)
    """
    metric = TreeDepthVariance()
    return metric(tree)


class TreeNumberOfRules:
    """
    Represents the Number of Rules metric
    """

    reference: float = 1
    name: str = "Number of Rules"

    def __call__(self, surrogate):
        return int(get_number_of_rules(surrogate))


def tree_number_of_rules(surrogate):
    """
    Calculates the number of rules in a decision tree surrogate model.
    Parameters
    ----------
        surrogate: A surrogate model, typically a decision tree, for which the number of rules is to be calculated.

    Returns
    -------
        int: The number of rules present in the surrogate model.

    Examples
    --------

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from holisticai.explainability.metrics import tree_number_of_rules
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = DecisionTreeClassifier()
    >>> clf.fit(X, y)
    >>> tree_number_of_rules(clf)
    """

    m = TreeNumberOfRules()
    return m(surrogate)


class TreeNumberOfFeatures:
    """
    Represents the Number of Features metric
    """

    reference: float = 1
    name: str = "Number of Features"

    def __call__(self, surrogate):
        features = get_features(surrogate)
        features_used = np.unique(features[features >= 0])
        F1 = len(features_used)
        return int(F1)


def tree_number_of_features(surrogate):
    m = TreeNumberOfFeatures()
    return m(surrogate)
