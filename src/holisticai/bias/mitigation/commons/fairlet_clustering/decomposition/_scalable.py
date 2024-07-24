from collections import defaultdict

import numpy as np
from holisticai.bias.mitigation.commons.fairlet_clustering.decomposition._vanilla import VanillaFairletDecomposition
from holisticai.utils.transformers.bias import SensitiveGroups

EPSILON = 0.0001


class TreeNode:
    """
    A node in the quadtree used for fairlet decomposition

    Attributes
    ----------
    children : list
        The children of the node.
    cluster : list
        The indices of the cluster.
    reds : list
        The indices of the red points.
    blues : list
        The indices of the blue points.

    Methods
    -------
    set_cluster(cluster)
        Sets the cluster of the node.
    add_child(child)
        Adds a child to the node.
    populate_colors(colors)
        Populates the red and blue points for each node.
    """

    def __init__(self):
        self.children = []

    def set_cluster(self, cluster):
        """
        Sets the cluster of the node.

        Parameters
        ----------
        cluster : list
            The indices of the cluster.
        """
        self.cluster = cluster

    def add_child(self, child):
        """
        Adds a child to the node.

        Parameters
        ----------
        child : TreeNode
            The child node to add.
        """
        self.children.append(child)

    def populate_colors(self, colors):
        """
        Populate auxiliary lists of red and blue points for each node, bottom-up

        Parameters
        ----------
        colors : list
            The colors of the points.
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
    A scalable version of the fairlet decomposition algorithm

    Parameters
    ----------
    p : int
        The number of red points in a fairlet.
    q : int
        The number of blue points in a fairlet.

    Attributes
    ----------
    fairlets : list
        The fairlets.
    fairlet_centers : list
        The centers of the fairlets.
    p : int
        The number of red points in a fairlet.
    q : int
        The number of blue points in a fairlet.
    _sensgroups : SensitiveGroups
        The sensitive groups.

    Methods
    -------
    decompose(node, dataset, donelist, depth)
        Decomposes the dataset into fairlets.
    fit_transform(dataset, group_a, group_b)
        Fits the dataset and transforms it into fairlets.
    balanced(p, q, reds, blues)
        Checks if the dataset is balanced.
    """

    def __init__(self, p, q):
        self.fairlets = []
        self.fairlet_centers = []
        assert p <= q, "Please use balance parameters in the correct order"
        self.p = p
        self.q = q
        self._sensgroups = SensitiveGroups()

    def decompose(self, node, dataset, donelist, depth):
        """
        Decomposes the dataset into fairlets.

        Parameters
        ----------
        node : TreeNode
            The node in the quadtree.
        dataset : list
            The dataset.
        donelist : list
            The list of points that have already been clustered.
        depth : int
            The depth of the node in the quadtree.

        Returns
        -------
        int
            The cost of the decomposition.
        """
        p = self.p
        q = self.q

        # Leaf
        if len(node.children) == 0:
            node.reds = [i for i in node.reds if donelist[i] == 0]
            node.blues = [i for i in node.blues if donelist[i] == 0]
            assert self.balanced(p, q, len(node.reds), len(node.blues)), "Reached unbalanced leaf"
            return super().decompose(node.blues, node.reds, dataset)

        # Preprocess children nodes to get rid of points that have already been clustered
        for child in node.children:
            child.reds = [i for i in child.reds if donelist[i] == 0]
            child.blues = [i for i in child.blues if donelist[i] == 0]

        R = [len(child.reds) for child in node.children]
        B = [len(child.blues) for child in node.children]

        if sum(R) == 0 or sum(B) == 0:
            if sum(R) == 0 and sum(B) == 0:
                raise ValueError("One color class became empty for this node while the other did not")
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

        return super().decompose(blues, reds, dataset) + sum(
            [self.decompose(child, dataset, donelist, depth + 1) for child in node.children]
        )

    def fit_transform(self, dataset, group_a, group_b):
        """
        Fits the dataset and transforms it into fairlets.

        Parameters
        ----------
        dataset : list
            The dataset.
        group_a : list
            The first group.
        group_b : list
            The second group.

        Returns
        -------
        list
            The fairlets.
        list
            The centers of the fairlets.
        float
            The cost of the decomposition.
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
        cost = self.decompose(root, dataset, donelist, 0)
        return self.fairlets, self.fairlet_centers, cost


### QUADTREE CODE ###


def build_quadtree(dataset, max_levels=0, random_shift=True):
    """
    Builds a quadtree for the given dataset. If max_levels=0 there no level limit,
    quadtree will partition until all clusters are singletons

    Parameters
    ----------
    dataset : numpy.ndarray
        The dataset.
    max_levels : int, optional
        The maximum number of levels in the quadtree. Default is 0.
    random_shift : bool, optional
        Whether to randomly shift the dataset. Default is True.

    Returns
    -------
    TreeNode
        The root of the quadtree.
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
    Builds a quadtree recursively to partition the dataset into clusters.

    Parameters
    ----------
    dataset : numpy.ndarray
        The input dataset.
    cluster : list
        The indices of the data points in the current cluster.
    lower : numpy.ndarray
        The lower bounds of the current partition.
    upper : numpy.ndarray
        The upper bounds of the current partition.
    max_levels : int
        The maximum number of levels to build the quadtree.
    shift : numpy.ndarray
        The shift vector to adjust the partition.

    Returns
    -------
    TreeNode:
        The root node of the quadtree.

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
