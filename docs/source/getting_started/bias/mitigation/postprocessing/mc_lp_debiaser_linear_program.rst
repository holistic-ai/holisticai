Linear Program Method for Multiclass
------------------------------------

.. note::
    **Learning tasks:** Multiclassification.

Introduction
~~~~~~~~~~~~
A previous linear programming technique is extended to accommodate a theoretically arbitrarily large number of discrete outcomes and levels of a protected attribute. This method aims to achieve fairness in multiclass settings by adjusting the predictions of a blackbox classifier to reduce disparity across different protected groups.

Description
~~~~~~~~~~~
- **Problem definition**

  The primary goal is to adjust the predictions of a blackbox classifier to ensure fairness across different protected groups while maintaining as much predictive performance as possible. The method focuses on reducing post-adjustment disparity in multiclass classification problems.

- **Main features**

  - Extends the linear programming technique to handle multiple discrete outcomes and protected attribute levels.
  - Evaluates the tradeoff between fairness and discrimination.

- **Step-by-step description of the approach**

  1. **Initial Setup**: 

     - Train a classifier to predict the multiclass outcome.
     - Obtain the initial predictions :math:`\hat{Y}`.

  2. **Linear Program Formulation**:

     - Define the fairness constraints based on the type of fairness desired (e.g., equalized odds, equal opportunity, demographic parity).
     - Formulate the linear program to adjust the predictions :math:`Y_{\text{adj}}` such that the fairness constraints are satisfied.

  3. **Solving the Linear Program**:

     - Solve the linear program on the entire dataset to obtain the adjusted probabilities :math:`P_a`.
     - Use these probabilities to generate the adjusted predictions :math:`Y_{\text{adj}}`.

Basic Usage
~~~~~~~~~~~~~~
You can find an example of using the Linear Program Method for Multiclass in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/multi_classification/demos/postprocessing.html#1.-LP-Debiaser-Multiclass>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.LPDebiaserMulticlass`.

References
~~~~~~~~~~~~~~
1. Putzel, Preston, and Scott Lee. "Blackbox Post-Processing for Multiclass Fairness."arXiv preprint arXiv:2201.04461 (2022);.
