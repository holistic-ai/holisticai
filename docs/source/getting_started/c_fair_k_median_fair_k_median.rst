**Fair k-Median Method**
========================

**Introduction**
----------------
The fair k-median method aims to address the issue of fairness in clustering by ensuring that the clustering solution does not disproportionately favor any particular group. This method is particularly relevant in scenarios where data points belong to different demographic groups, and it is crucial to provide equitable representation and minimize bias in the clustering process.

**Description**
---------------
The fair k-median method involves several key steps to ensure fairness in clustering:

1. **Bundling**: For each point :math:`u \in S`, a bundle :math:`B_u` is created, which consists of the centers that exclusively serve :math:`u`. During the rounding procedure, each bundle :math:`B_u` is treated as a single entity, where at most one center from it will be opened. The probability of opening a center from a bundle, :math:`B_u`, is the sum of :math:`y_c` for all :math:`c \in B_u`, referred to as the bundle's volume.

2. **Matching**: The generated bundles have the property that their volume lies within the range :math:`[1/2, 1]`. Therefore, given any two bundles, at least one center from them should be opened. While there are at least two unmatched points in :math:`S`, the corresponding bundles of the two closest unmatched points in :math:`S` are matched.

3. **Sampling**: Given the matching generated in the previous phase, the algorithm iterates over its members and considers the bundle volumes as probabilities to open :math:`k` centers in expectation. The centers picked in the sampling phase are returned as the final :math:`k` centers.

**Equations/Algorithms**
------------------------

**LS-Fair k-Median Algorithm**

The LS-Fair k-median algorithm is a heuristic local search algorithm that modifies the standard local search algorithm to minimize the maximum average cost over all groups. The algorithm is presented as follows:

.. math::
    :label: ls-fair-k-median

    \begin{align*}
    &\text{LS-Fair k-median}(k, \text{cost}, X, X_1, \ldots, X_m) \\
    &S \leftarrow \text{an arbitrary set of } k \text{ centers from } X \\
    &\text{old\_cost} \leftarrow \infty \\
    &\text{new\_cost} \leftarrow \max(\text{cost}_S(X_1), \ldots, \text{cost}_S(X_m)) \\
    &\text{while there is } t' \in X \text{ and } t \in S \text{ such that } \\
    &\max(\text{cost}_{S \setminus t \cup t'}(X_1), \ldots, \text{cost}_{S \setminus t \cup t'}(X_m)) < \text{old\_cost} \text{ do} \\
    &\quad S \leftarrow S \setminus t \cup t' \\
    &\quad \text{old\_cost} \leftarrow \max(\text{cost}_{S \setminus t \cup t'}(X_1), \ldots, \text{cost}_{S \setminus t \cup t'}(X_m)) \\
    &\text{return } S
    \end{align*}

**Usage Examples**
------------------
The fair k-median method was tested on several datasets to evaluate its performance:

- **Census Dataset**

- **Bank Dataset**

The results demonstrated the effectiveness of the fair k-median method, particularly in cases where the groups had different distributions.

**Advantages and Limitations**
------------------------------

*Advantages:*

- Ensures fair representation of different groups in the clustering solution.
- Reduces bias in the clustering process, particularly in datasets with imbalanced group sizes.
- Provides a systematic approach to handle fairness in clustering.

*Limitations:*

- The LS-Fair k-median algorithm may have local optima that are arbitrarily worse than the global optimum.
- The method may be computationally intensive due to the additional steps involved in ensuring fairness.
- Theoretical bounds are not provided for the LS-Fair k-median algorithm, which may affect its reliability in certain scenarios.

**References**
---------------
1. Abbasi, Mohsen, Aditya Bhaskara, and Suresh Venkatasubramanian. "Fair clustering via equitable group representations." Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency. 2021.