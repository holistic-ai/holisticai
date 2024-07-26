import logging
from collections import defaultdict

import numpy as np
from holisticai.bias.mitigation.commons.fairlet_clustering.decompositions._vanilla import VanillaFairletDecomposition
from holisticai.utils.transformers.bias import SensitiveGroups

logger = logging.getLogger(__name__)

EPSILON = 0.0001


class TreeNode:
    """
    A node in the quadtree used for scalable fairlet decomposition

    Attributes
    ----------
    children : list
        List of children nodes
    cluster : list
        List of indices of points in the cluster
    reds : list
        List of indices of red points in the cluster
    blues : list
        List of indices of blue points in the cluster

    Methods
    -------
    set_cluster(cluster)
        Set the cluster attribute
    add_child(child)
        Add a child node
    populate_colors(colors)
        Populate the reds and blues lists for each node
    """

    def __init__(self):
        self.children = []

    def set_cluster(self, cluster):
        """
        Set the cluster attribute

        Parameters
        ----------
        cluster : list
            List of indices of points in the cluster
        """
        self.cluster = cluster

    def add_child(self, child):
        """
        Add a child node

        Parameters
        ----------
        child : TreeNode
            Child node to add
        """
        self.children.append(child)

    def populate_colors(self, colors):
        """
        Populate auxiliary lists of red and blue points for each node, bottom-up

        Parameters
        ----------
        colors : list
            List of colors for each point in the dataset
        """
        self.reds = []
        self.blues = []
        if len(self.children) == 0:
            # Leaf
            for i in self.cluster:
                if colors[i] == 0:
                    self.reds.append(i)
                else:
                    self.blues.append(i)
        else:
            # Not a leaf
            for child in self.children:
                child.populate_colors(colors)
                self.reds.extend(child.reds)
                self.blues.extend(child.blues)


class ScalableFairletDecomposition(VanillaFairletDecomposition):
    """
    Scalable fairlet decomposition algorithm

    Parameters
    ----------
    p : float
        Balance parameter
    q : float
        Balance parameter

    Attributes
    ----------
    fairlets : list
        List of fairlets
    fairlet_centers : list
        List of fairlet centers
    _sensgroups : SensitiveGroups
        Sensitive groups transformer

    Methods
    -------
    fit_transform(dataset, group_a, group_b)
        Fit the model and transform the dataset
    balanced(p, q, R, B)
        Check if a set of points is balanced
    """

    def __init__(self, p, q):
        self.fairlets = []
        self.fairlet_centers = []
        assert p <= q, "Please use balance parameters in the correct order"
        self.p = p
        self.q = q
        self._sensgroups = SensitiveGroups()

    def _decompose(self, node, dataset, donelist, depth):
        """
        Recursively decompose a node in the quadtree

        Parameters
        ----------
        node : TreeNode
            Node to decompose
        dataset : np.ndarray
            Dataset
        donelist : list
            List of points that have already been clustered
        depth : int
            Depth of the node in the quadtree

        Returns
        -------
        float
            Cost of the decomposition
        """
        p = self.p
        q = self.q

        # Leaf
        if len(node.children) == 0:
            node.reds = [i for i in node.reds if donelist[i] == 0]
            node.blues = [i for i in node.blues if donelist[i] == 0]
            assert self.balanced(p, q, len(node.reds), len(node.blues)), "Reached unbalanced leaf"
            return super()._decompose(node.blues, node.reds, dataset)

        # Preprocess children nodes to get rid of points that have already been clustered
        for child in node.children:
            child.reds = [i for i in child.reds if donelist[i] == 0]
            child.blues = [i for i in child.blues if donelist[i] == 0]

        R = [len(child.reds) for child in node.children]
        B = [len(child.blues) for child in node.children]

        if sum(R) == 0 or sum(B) == 0:
            logger.warning("One color class became empty for this node while the other did not")
            return 0

        NR = 0
        NB = 0

        # Phase 1: Add must-remove nodes
        for i in range(len(node.children)):
            if R[i] >= B[i]:
                must_remove_red = max(0, R[i] - int(np.floor(B[i] * q * 1.0 / p)))
                R[i] -= must_remove_red
                NR += must_remove_red
            else:
                must_remove_blue = max(0, B[i] - int(np.floor(R[i] * q * 1.0 / p)))
                B[i] -= must_remove_blue
                NB += must_remove_blue

        # Calculate how many points need to be added to smaller class until balance
        if NR >= NB:
            # Number of missing blues in (NR,NB)
            missing = max(0, int(np.ceil(NR * p * 1.0 / q)) - NB)
        else:
            # Number of missing reds in (NR,NB)
            missing = max(0, int(np.ceil(NB * p * 1.0 / q)) - NR)

        # Phase 2: Add may-remove nodes until (NR,NB) is balanced or until no more such nodes
        for i in range(len(node.children)):
            if missing == 0:
                assert self.balanced(p, q, NR, NB), "Something went wrong"
                break
            if NR >= NB:
                may_remove_blue = B[i] - int(np.ceil(R[i] * p * 1.0 / q))
                remove_blue = min(may_remove_blue, missing)
                B[i] -= remove_blue
                NB += remove_blue
                missing -= remove_blue
            else:
                may_remove_red = R[i] - int(np.ceil(B[i] * p * 1.0 / q))
                remove_red = min(may_remove_red, missing)
                R[i] -= remove_red
                NR += remove_red
                missing -= remove_red

        # Phase 3: Add unsatuated fairlets until (NR,NB) is balanced
        for i in range(len(node.children)):
            if self.balanced(p, q, NR, NB):
                break
            if R[i] >= B[i]:
                num_saturated_fairlets = int(R[i] / q)
                excess_red = R[i] - q * num_saturated_fairlets
                excess_blue = B[i] - p * num_saturated_fairlets
            else:
                num_saturated_fairlets = int(B[i] / q)
                excess_red = R[i] - p * num_saturated_fairlets
                excess_blue = B[i] - q * num_saturated_fairlets
            R[i] -= excess_red
            NR += excess_red
            B[i] -= excess_blue
            NB += excess_blue

        if self.balanced(p, q, NR, NB):
            raise ValueError("Constructed node sets are unbalanced")

        reds = []
        blues = []
        for i in range(len(node.children)):
            for j in node.children[i].reds[R[i] :]:
                reds.append(j)
                donelist[j] = 1
            for j in node.children[i].blues[B[i] :]:
                blues.append(j)
                donelist[j] = 1

        if len(reds) == NR and len(blues) == NB:
            raise ValueError("Something went horribly wrong")

        return super()._decompose(blues, reds, dataset) + sum(
            [self._decompose(child, dataset, donelist, depth + 1) for child in node.children]
        )

    def fit_transform(self, dataset, group_a, group_b):
        """
        Fit the model and transform the dataset

        Parameters
        ----------
        dataset : np.ndarray
            Dataset
        group_a : np.ndarray
            Group membership vector (binary)
        group_b : np.ndarray
            Group membership vector (binary)

        Returns
        -------
        tuple
            Tuple containing fairlets, fairlet centers, and cost
        """
        sensitive_groups = np.c_[group_a, group_b]
        colors = self._sensgroups.fit_transform(sensitive_groups, convert_numeric=True)
        root = build_quadtree(dataset)
        "Main fairlet clustering function, returns cost wrt original metric (not tree metric)"
        p = self.p
        q = self.q
        root.populate_colors(colors)
        assert self.balanced(p, q, len(root.reds), len(root.blues)), "Dataset is unbalanced"
        root.populate_colors(colors)
        donelist = [0] * dataset.shape[0]
        cost = self._decompose(root, dataset, donelist, 0)
        return self.fairlets, self.fairlet_centers, cost


def build_quadtree(dataset, max_levels=0, random_shift=True):
    """
    If max_levels=0 there no level limit, quadtree will partition until all clusters are singletons

    Parameters
    ----------
    dataset : np.ndarray
        Dataset
    max_levels : int, optional
        Maximum depth of the quadtree. Default is 0.
    random_shift : bool, optional
        Whether to randomly shift the dataset. Default is True.

    Returns
    -------
    TreeNode
        Root node of the quadtree
    """
    dimension = dataset.shape[1]
    lower = np.amin(dataset, axis=0)
    upper = np.amax(dataset, axis=0)

    shift = np.zeros(dimension)
    if random_shift:
        for d in range(dimension):
            spread = upper[d] - lower[d]
            shift[d] = np.random.uniform(0, spread)
            upper[d] += spread

    return build_quadtree_aux(dataset, range(dataset.shape[0]), lower, upper, max_levels, shift)


def build_quadtree_aux(dataset, cluster, lower, upper, max_levels, shift):
    """
    Recursive helper function for building a quadtree

    Parameters
    ----------
    dataset : np.ndarray
        Dataset
    cluster : list
        List of indices of points in the cluster
    lower : np.ndarray
        Lower bounds of the cell
    upper : np.ndarray
        Upper bounds of the cell
    max_levels : int
        Maximum depth of the quadtree
    shift : np.ndarray
        Shift vector

    Returns
    -------
    TreeNode
        Root node of the quadtree
    """

    dimension = dataset.shape[1]
    cell_too_small = True
    for i in range(dimension):
        if upper[i] - lower[i] > EPSILON:
            cell_too_small = False

    node = TreeNode()
    if max_levels == 1 or len(cluster) <= 1 or cell_too_small:
        # Leaf
        node.set_cluster(cluster)
        return node

    # Non-leaf
    midpoint = 0.5 * (lower + upper)
    subclusters = defaultdict(list)
    for i in cluster:
        subclusters[tuple([dataset[i, d] + shift[d] <= midpoint[d] for d in range(dimension)])].append(i)
    for edge, subcluster in subclusters.items():
        sub_lower = np.zeros(dimension)
        sub_upper = np.zeros(dimension)
        for d in range(dimension):
            if edge[d]:
                sub_lower[d] = lower[d]
                sub_upper[d] = midpoint[d]
            else:
                sub_lower[d] = midpoint[d]
                sub_upper[d] = upper[d]
        node.add_child(build_quadtree_aux(dataset, subcluster, sub_lower, sub_upper, max_levels - 1, shift))
    return node
