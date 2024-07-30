Fair k-Median Method
--------------------

.. note::
    **Learning tasks:** Clustering.

Introduction
~~~~~~~~~~~~~~~~
The fair k-median method aims to address the issue of fairness in clustering by ensuring that the clustering solution does not disproportionately favor any particular group. This method is particularly relevant in scenarios where data points belong to different demographic groups, and it is crucial to provide equitable representation and minimize bias in the clustering process.

Description
~~~~~~~~~~~~~~~~
The fair k-median method involves several key steps to ensure fairness in clustering:

1. **Bundling**: For each point :math:`u \in S`, a bundle :math:`B_u` is created, which consists of the centers that exclusively serve :math:`u`. During the rounding procedure, each bundle :math:`B_u` is treated as a single entity, where at most one center from it will be opened. The probability of opening a center from a bundle, :math:`B_u`, is the sum of :math:`y_c` for all :math:`c \in B_u`, referred to as the bundle's volume.

2. **Matching**: The generated bundles have the property that their volume lies within the range :math:`[1/2, 1]`. Therefore, given any two bundles, at least one center from them should be opened. While there are at least two unmatched points in :math:`S`, the corresponding bundles of the two closest unmatched points in :math:`S` are matched.

3. **Sampling**: Given the matching generated in the previous phase, the algorithm iterates over its members and considers the bundle volumes as probabilities to open :math:`k` centers in expectation. The centers picked in the sampling phase are returned as the final :math:`k` centers.

References
~~~~~~~~~~~~~~~~
1. Abbasi, Mohsen, Aditya Bhaskara, and Suresh Venkatasubramanian. "Fair clustering via equitable group representations." Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency. 2021.