Fair k-Center Clustering Method
----------------

.. note::
    **Learning tasks:** Clustering.

Introduction
~~~~~~~~~~~~~~~~
The Fair k-Center Clustering method addresses the problem of centroid-based clustering, such as k-center, in a way that ensures fair representation of different demographic groups. This method is particularly relevant in scenarios where the data set comprises multiple demographic groups, and there is a need to select a fixed number of representatives (centers) from each group to form a summary. The method extends the traditional k-center clustering problem by incorporating fairness constraints, ensuring that each group is fairly represented in the selected centers.

Description
~~~~~~~~~~~~~~~~
The Fair k-Center Clustering method aims to minimize the maximum distance between any data point and its closest center while ensuring that a specified number of centers are chosen from each demographic group. 

The method involves a recursive algorithm that handles the fairness constraints by iteratively selecting centers and ensuring the required representation from each group. The algorithm can be broken down into the following steps:

1. **Initialization**: Start with an empty set of centers and the given parameters.
2. **Center Selection**: Use a greedy strategy to select centers that maximize the distance to the current set of centers.
3. **Fairness Adjustment**: Adjust the selected centers to ensure the required number of centers from each group.
4. **Recursion**: If the fairness constraints are not met, recursively apply the algorithm to a subset of the data until the constraints are satisfied.

References
~~~~~~~~~~~~~~~~
1. Kleindessner, Matthäus, Pranjal Awasthi, and Jamie Morgenstern. "Fair k-center clustering for data summarization." International Conference on Machine Learning. PMLR, 2019.