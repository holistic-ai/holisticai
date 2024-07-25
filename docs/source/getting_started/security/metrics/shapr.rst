SHAPr: SHapley vAlue-based Privacy Risk
=======================================

Definition
----------

SHAPr is a membership privacy metric based on Shapley values, originally intended to measure the contribution of a training data record on model utility. SHAPr approximates a leave-one-out (LOO) technique to estimate the membership privacy risk of individual training data records in machine learning models.

Mathematical Formulation
-------------------------

The Shapley value :math:`(\phi_i)` of a data record :math:`(z_i = (x_i, y_i))` is defined as:

.. math::

    \phi_i = \frac{1}{|D_{tr}|} \sum_{S \subseteq D_{tr} \setminus \{z_i\}} \frac{1}{{|D_{tr} - 1| \choose |S|}} \left[ U(S \cup \{z_i\}) - U(S) \right]

where \(S\) is a randomly chosen subset of :math:`D_{tr} \setminus \{z_i\}` and \(U(S)\) is the utility of the model trained on subset \(S\).

Interpretation
--------------

- **Positive SHAPr Score \(\phi > 0\):** Indicates that the data record contributed positively to the model's utility, suggesting a higher likelihood of memorization and, therefore, a higher membership privacy risk.
- **Negative SHAPr Score \(\phi < 0\):** Indicates that the data record was either harmful or not significantly contributing to the model's utility, suggesting lower susceptibility to membership inference attacks.
- **Zero SHAPr Score \(\phi = 0\):** Implies that the presence or absence of the data record does not affect the model's utility, indicating no membership privacy risk.

Reference
---------

For more details, see the paper `SHAPr: An Efficient and Versatile Membership Privacy Risk Metric <https://arxiv.org/abs/2112.02230>`_.
