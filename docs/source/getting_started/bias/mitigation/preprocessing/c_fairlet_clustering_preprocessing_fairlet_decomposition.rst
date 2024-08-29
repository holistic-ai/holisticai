Fairlet Decomposition
---------------------

.. note::
    **Learning tasks:** Clustering.

Introduction
~~~~~~~~~~~~
The preprocessing (r, b)-fairlet decomposition method is designed to partition a set of points into smaller, balanced subsets called fairlets. This method is particularly useful in fair clustering, where the goal is to ensure that each cluster has a balanced representation of different groups (e.g., red and blue points). The method leverages a hierarchical structure to efficiently compute these fairlets, ensuring that the resulting clusters are balanced according to specified parameters.

Description
~~~~~~~~~~~
- **Problem definition**

  The problem addressed by the (r, b)-fairlet decomposition method is to partition a set of points into smaller subsets (fairlets) such that each subset is balanced according to the given parameters :math:`r` and :math:`b`. Specifically, a subset is considered (r, b)-balanced if the ratio of the number of red points to blue points in each subset is at least :math:`\frac{b}{r}`. The goal is to minimize the total number of points that need to be removed to achieve this balance.

- **Main features**

  The main features of the (r, b)-fairlet decomposition method include:
  
  - Efficiently partitioning points into balanced fairlets.
  - Minimizing the number of points removed to achieve balance.
  - Utilizing a hierarchical structure (HST) to facilitate the decomposition process.
  - Ensuring that the resulting fairlets are balanced according to the specified parameters.

- **Step-by-step description of the approach**

  Given a node :math:`v` which is a leaf node of the tree :math:`T`, an arbitrary (r, b)-fairlet decomposition of the points in :math:`T(v)` is returned.

  1. **Minimize Heavy Points**:

     - For each non-empty child :math:`i` of :math:`v`, compute the number of red and blue points :math:`\{N_i^r, N_i^b\}`.
     - Use a function to approximately minimize the total number of heavy points with respect to :math:`v`.

  2. **Decompose Heavy Points**:

     - Initialize an empty set :math:`P_v`.
     - For each non-empty child :math:`i` of :math:`v`, remove an arbitrary set of :math:`x_i^r` red and :math:`x_i^b` blue points from :math:`T(v_i)` and add them to :math:`P_v`.
     - Output an arbitrary (r, b)-fairlet decomposition of points in :math:`P_v`.

Basic Usage
~~~~~~~~~~~~~~

You can find an example of using the Fairlet Decomposition method in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/clustering/demos/preprocessing.html#1.-Fairlet>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.FairletClusteringPreprocessing`.

References
~~~~~~~~~~~~~~
1. Backurs, Arturs, et al. "Scalable fair clustering." International Conference on Machine Learning. PMLR, 2019.