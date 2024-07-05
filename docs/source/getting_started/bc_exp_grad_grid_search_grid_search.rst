**Grid Search**
=================

**Introduction**
----------------
Grid search is a method used to select a deterministic classifier from a set of candidate classifiers obtained from the saddle point of a Lagrangian function. This method is particularly useful when the number of constraints is small, such as in demographic parity or equalized odds with a binary protected attribute. The goal is to find a classifier that balances the tradeoff between accuracy and fairness.

**Description**
---------------
Grid search involves the following steps:

1. **Candidate Classifiers**: A set of candidate classifiers is obtained from the saddle point :math:`(Q^\dagger, \lambda^\dagger)`. Since :math:`Q^\dagger` is a minimizer of :math:`L(Q, \lambda^\dagger)` and :math:`L` is linear in :math:`Q` the distribution :math:`Q^\dagger` puts non-zero mass only on classifiers that are the Q-player’s best responses to :math:`\lambda^\dagger`.
2. **Best Response Calculation**: If :math:`\lambda^\dagger` is known, one can retrieve a best response via a reduction to cost-sensitive learning.
3. **Grid Search**: When the number of constraints is small, a grid of values for :math:`\lambda` is considered. For each value, the best response is calculated, and the value with the desired tradeoff between accuracy and fairness is selected.

**Equations/Algorithms**
------------------------
For demographic parity with a binary protected attribute :math:`A \in \{a, a'\}` the grid search can be conducted in a single dimension. The reduction takes two real-valued arguments :math:`\lambda_a` and :math:`\lambda_{a'}`, and adjusts the costs for predicting :math:`h(X_i) = 1` by the amounts:

.. math::
    :label: dp-adjustments

    \delta_a = \frac{\lambda_a}{p_a} - \lambda_a - \lambda_{a'} \\
    \delta_{a'} = \frac{\lambda_{a'}}{p_{a'}} - \lambda_a - \lambda_{a'}

These adjustments satisfy :math:`p_a \delta_a + p_{a'} \delta_{a'} = 0`, so instead of searching over :math:`\lambda_a` and :math:`\lambda_{a'}`, the grid search can be conducted over :math:`\delta_a` alone, applying the adjustment :math:`\delta_{a'} = -\frac{p_a \delta_a}{p_{a'}}` to the protected attribute value :math:`a'`.

For equalized odds with a binary protected attribute :math:`A \in \{a, a'\}`, the adjustments are:

.. math::
    :label: eo-adjustments

    \delta(a, y) = \frac{\lambda(a, y)}{p(a, y)} - \frac{\lambda(a, y) + \lambda(a', y)}{p(\ast, y)}

These adjustments satisfy :math:`p(a, y) \delta(a, y) + p(a', y) \delta(a', y) = 0`, so the grid search can be conducted over :math:`\delta(a, 0)` and :math:`\delta(a, 1)`, setting the parameters for :math:`a'` to :math:`\delta(a', y) = -\frac{p(a, y) \delta(a, y)}{p(a', y)}`.

**Usage Examples**
------------------
- **Demographic Parity (DP)**: When the protected attribute is binary, e.g., :math:`A \in \{a, a'\}`, the grid search can be conducted in a single dimension. For example, in the adult income dataset, the task is to predict whether someone makes more than $50k per year, with gender as the protected attribute.
- **Equalized Odds (EO)**: For the COMPAS recidivism dataset, the task is to predict recidivism with race as the protected attribute (restricted to white and black defendants). The grid search can be conducted over :math:`\delta(a, 0)` and :math:`\delta(a, 1)`.

**Advantages and Limitations**
------------------------------
*Advantages:*

- Simple and intuitive method for selecting a deterministic classifier.
- Effective when the number of constraints is small.
- Allows for a clear tradeoff between accuracy and fairness.

*Limitations:*

- Not feasible for non-binary protected attributes due to the high dimensionality of the grid search.
- Computationally expensive as the number of constraints increases.
- May not achieve the lowest disparities on the training data compared to other methods.

**References**
---------------
1. Agarwal, A., Beygelzimer, A., Dudik, M., Langford, J., & Wallach, H. (2018). A reductions approach to fair classification. In Advances in Neural Information Processing Systems (pp. 656-666).