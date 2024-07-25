Fairlet Decomposition Method
------------------------------

.. note::
    **Learning tasks:** Clustering.

Introduction
~~~~~~~~~~~~~~~~
The Fairlet Decomposition method is a scalable algorithm designed to partition a set of points into fairlets, which are small clusters that maintain a specified balance between two types of points (e.g., red and blue points). This method is particularly useful in fair clustering tasks where the goal is to ensure that each cluster has a balanced representation of different types of points.

Description
~~~~~~~~~~~~~~~~
The Fairlet Decomposition method operates on a hierarchical space tree (HST) embedding of the input points. The main goal is to partition the points into fairlets that are balanced according to given parameters :math:`(r, b)`, where :math:`r` and :math:`b` denote the number of red and blue points, respectively.

The method consists of three main steps:

1. **Approximately Minimize the Total Number of Heavy Points:**

   - For each non-empty child of a node :math:`v` in the HST, compute the number of red and blue points that need to be removed to make the remaining points :math:`(r, b)`-balanced.
   - Solve the Minimum Heavy Points Problem to find the optimal number of points to remove.

2. **Find an (r, b)-Fairlet Decomposition of Heavy Points:**

   - Remove the computed number of red and blue points from each child and add them to a set :math:`P_v`.
   - Output an :math:`(r, b)`-fairlet decomposition of the points in :math:`P_v`.

3. **Proceed to the Children of the Node:**

   - Recursively apply the Fairlet Decomposition method to each non-empty child of the node.

References
~~~~~~~~~~~~~~~~
1. Backurs, Arturs, et al. "Scalable fair clustering." International Conference on Machine Learning. PMLR, 2019.