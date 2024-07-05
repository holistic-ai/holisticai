**Fairlet Decomposition Method**
===============================

**Introduction**
----------------
The Fairlet Decomposition method is a scalable algorithm designed to partition a set of points into fairlets, which are small clusters that maintain a specified balance between two types of points (e.g., red and blue points). This method is particularly useful in fair clustering tasks where the goal is to ensure that each cluster has a balanced representation of different types of points.

**Description**
---------------
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

**Equations/Algorithms**
------------------------

.. math::
    :label: algorithm-fairlet-decomposition

    \begin{align*}
    &\text{FairletDecomposition}(v, r, b):\text{returns an } (r, b)\text{-fairlet decomposition of the points in } T(v)\\
    1: & \text{ if } v \text{ is a leaf node of } T \text{ then} \\
    2: & \quad \text{return an arbitrary } (r, b)\text{-fairlet decomposition of the points in } T(v) \\
    3: & \text{ end if} \\
    4: & \{x_i^r, x_i^b\}_i \leftarrow \text{MinHeavyPoints}(\{N_i^r, N_i^b\}_{i \in [\gamma d]}, r, b) \\
    5: & P_v \leftarrow \emptyset \\
    6: & \text{ for all non-empty children } i \in [\gamma d] \text{ of } v \text{ do} \\
    7: & \quad \text{remove an arbitrary set of } x_i^r \text{ red and } x_i^b \text{ blue points from } T(v_i) \text{ and add them to } P_v \\
    8: & \text{ end for} \\
    9: & \text{output an } (r, b)\text{-fairlet decomposition of } P_v \\
    10: & \text{ for all non-empty children } i \in [\gamma d] \text{ of } v \text{ do} \\
    11: & \quad \text{FairletDecomposition}(v_i, r, b) \\
    12: & \text{ end for}
    \end{align*}

**Usage Examples**
------------------
The Fairlet Decomposition method was tested on various datasets to demonstrate its effectiveness in creating balanced clusters. For instance, the method was applied to the Diabetes and Bank datasets, resulting in a balanced fairlet decomposition.

**Advantages and Limitations**
------------------------------

*Advantages:*

- The method ensures that each cluster is balanced according to the specified parameters.
- It operates in near-linear time, making it scalable for large datasets.
- The approach does not disregard any part of the input, potentially leading to better solutions.

*Limitations:*

- The method guarantees an approximation factor that is logarithmic in the number of points, which may not be optimal in all cases.
- The algorithm's performance depends on the quality of the HST embedding, which may introduce additional complexity.

**References**
---------------
1. Backurs, Arturs, et al. "Scalable fair clustering." International Conference on Machine Learning. PMLR, 2019.