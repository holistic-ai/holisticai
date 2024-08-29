Grid Search
------------

.. note::
    **Learning tasks:** Binary classification, regression.

Introduction
~~~~~~~~~~~~
Grid search is a method used to select a deterministic classifier from a set of candidate classifiers obtained from the saddle point of a Lagrangian function. This method is particularly useful when the number of constraints is small, such as in demographic parity or equalized odds with a binary protected attribute. The goal is to find a classifier that balances the tradeoff between accuracy and fairness.

Description
~~~~~~~~~~~
Grid search involves the following steps:

1. **Candidate Classifiers**: A set of candidate classifiers is obtained from the saddle point :math:`(Q^\dagger, \lambda^\dagger)`. Since :math:`Q^\dagger` is a minimizer of :math:`L(Q, \lambda^\dagger)` and :math:`L` is linear in :math:`Q` the distribution :math:`Q^\dagger` puts non-zero mass only on classifiers that are the Q-player’s best responses to :math:`\lambda^\dagger`.
2. **Best Response Calculation**: If :math:`\lambda^\dagger` is known, one can retrieve a best response via a reduction to cost-sensitive learning.
3. **Grid Search**: When the number of constraints is small, a grid of values for :math:`\lambda` is considered. For each value, the best response is calculated, and the value with the desired tradeoff between accuracy and fairness is selected.

Basic Usage
~~~~~~~~~~~~~~

You can find an example of using the Grid Search Reduction method in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/binary_classification/demos/inprocessing.html#3.-Grid-Search-Reduction>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.GridSearchReduction`.

References
~~~~~~~~~~
1. Agarwal, A., Beygelzimer, A., Dudik, M., Langford, J., & Wallach, H. (2018). A reductions approach to fair classification. In Advances in Neural Information Processing Systems (pp. 656-666).