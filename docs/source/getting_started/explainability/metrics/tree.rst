.. default-role:: math

Tree Based Metrics
===================

Decision trees are explainable models that provide insights into the decision-making process. Tree-based metrics evaluate the complexity and explainability of decision trees, helping to understand the model's behavior and performance.

Here, we describe the metrics used in holisticai to evaluate tree-based models.

.. contents:: Table of Contents
   :local:
   :depth: 1

What is a tree-based model?
---------------------------

A tree-based model is a type of supervised learning algorithm that makes decisions by splitting the data into subsets based on the values of the input features. The model uses a tree structure to represent the decision-making process, where each internal node represents a decision based on a feature, and each leaf node represents the output or prediction.

These models are widely used in various domains due to their interpretability and ease of use. Decision trees, random forests, and gradient boosting machines are popular tree-based models with many applications in research and industry.

Tree-Based Metrics
------------------

Tree-based metrics evaluate the complexity and explainability of decision trees. These metrics help quantify the model's behavior and performance. 


Weighted Average Depth (WAD)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This metric weighs the depth of each leaf by the number of points in its associated cluster. To minimize it, large clusters should be associated with shallower leaves (shorter explanations).

For a partition :math:`\mathcal{P}=(C_{1}, C_{2},\dots, C_{k})` induced by a binary decision tree :math:`\mathcal{D}` with :math:`k` leaves, where the cluster :math:`C_{i}` is associated with the leaf :math:`i`,

.. math::

    WAD(\mathcal{D}) = \frac{\sum\limits_{i=1}^{k}\mid C_{i}\mid l_{i}}{n} 

where :math:`l_{i}` is the number of conditions in the path from the root to leaf :math:`i`.


Weighted Average Explanation Size (WAES)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This metric replaces the depth of a leaf in WAD with the number of non-redundant tests in the path from root to the leaf.

For a partition :math:`\mathcal{P}=(C_{1}, C_{2},\dots, C_{k})` induced by a binary decision tree :math:`\mathcal{D}` with :math:`k` leaves, where the cluster :math:`C_{i}` is associated with the leaf :math:`i`,

.. math::

    WAES(\mathcal{D}) = \frac{\sum\limits_{i=1}^{k}\mid C_{i}\mid l_{i}^{nr}}{n} 

where :math:`l_{i}^{nr}` is the number of *non-redundant conditions* in the path from the root to leaf :math:`i`.


Weighted Gini Node Impurity (WGNI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Gini index is a measure of the impurity of a node, with a lower Gini index indicating a better split. This can be extended to a weighted Gini index across all nodes, which gives a sense of how well the tree is splitting the data overall.

For a node :math:`t`, the Gini index is given by:

.. math::

    GNI = 1 - \sum_{i=1}^{C} p_{i}^{2}

where :math:`p_{i}` is the proportion of samples of class :math:`i` in node :math:`t`, and :math:`C` is the number of classes. For the entire tree, the weighted Gini index can be computed as:

.. math::

    WGNI = \sum_{t \in \text{leaves}} \frac{n_{t}}{N} G(t)

where :math:`n_{t}` is the number of samples in node :math:`t`, and :math:`N` is the total number of samples in the tree.

The Gini index measures node impurity, with values ranging from 0 to 1: a lower value indicates a better split. The weighted Gini index provides an overall measure of how well the tree is splitting the data.

Tree Depth Variance (TDV)
~~~~~~~~~~~~~~~~~~~~~~~~~

Tree depth variance measures how uniformly the tree's leaves are distributed in terms of depth. A high variance indicates that some branches are much deeper than others, which can signal overfitting or poor generalization, diminishing the explainability over different clusters.

For a set of leaf depths :math:`\{d_1, d_2, \ldots, d_n\}`, the variance is calculated as:

.. math::

    TDV = \frac{1}{n} \sum_{i=1}^{n} (d_i - \mu_d)^2

where :math:`\mu_d` is the mean depth.

Tree Number of Features (TNF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A higher number of features may indicate a more complex model, which can affect the explainability and interpretability of the model.

The number of features used in the tree can be calculated as:

.. math::

    TNF = \sum_{t \in \text{leaves}} \text{features}(t)

where :math:`\text{features}(t)` is the number of features used in node :math:`t`.

Tree Number of Rules (TNR)
~~~~~~~~~~~~~~~~~~~~~~~~~

A higher number of rules may indicate a more complex model, which can affect the explainability and interpretability of the model.

The number of rules in the tree can be calculated as:

.. math::

    TNR = \sum_{t \in \text{leaves}} \text{rules}(t)

where :math:`\text{rules}(t)` is the number of rules used in node :math:`t`.