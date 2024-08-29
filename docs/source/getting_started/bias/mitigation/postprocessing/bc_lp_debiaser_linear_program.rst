Optimal Equalized Odds Predictor (Linear Programming Debiaser)
-----------------

.. note::
    **Learning tasks:** Binary classification, Regression.

Introduction
~~~~~~~~~~~~~~~
The optimal equalized odds predictor (Linear Programming Debiaser) is a method designed to ensure fairness in binary classification tasks by equalizing the odds across different groups defined by a protected attribute. This method is significant in mitigating biases in machine learning models, ensuring that the prediction outcomes are not disproportionately favorable or unfavorable to any particular group. The method leverages linear programming to derive a predictor that satisfies the equalized odds constraint, which requires that the true positive rate and false positive rate are the same across all groups.

Description
~~~~~~~~~~~~~~

- **Problem definition**

  The goal is to derive a predictor :math:`\tilde{Y}` from an initial predictor :math:`\hat{Y}` and a protected attribute :math:`A` such that the derived predictor satisfies the equalized odds constraint. This means that for all values of the protected attribute :math:`A`, the true positive rate and false positive rate should be equal. Formally, the problem can be defined as:

  .. math::
      \min_{\tilde{Y}} \mathbb{E}[\ell(\tilde{Y}, Y)]

  subject to:

  .. math::
      \forall a \in \{0, 1\}: \gamma_a(\tilde{Y}) \in P_a(\hat{Y}) \quad \text{(derived)}

  .. math::
      \gamma_0(\tilde{Y}) = \gamma_1(\tilde{Y}) \quad \text{(equalized odds)}

- **Main features**

  The main features of the optimal equalized odds predictor include:
  
  1. **Fairness Constraint**: Ensures that the predictor satisfies the equalized odds constraint, making the prediction fair across different groups.
  2. **Linear Programming**: Utilizes a linear programming approach to find the optimal predictor.
  3. **Derived Predictor**: The predictor is derived from an initial predictor and the protected attribute.

- **Step-by-step description of the approach**

  1. **Formulate the Optimization Problem**:
     
     The optimization problem is formulated to minimize the expected loss :math:`\mathbb{E}[\ell(\tilde{Y}, Y)]` subject to the constraints that ensure equalized odds. The constraints are defined such that the derived predictor :math:`\tilde{Y}` has the same true positive and false positive rates across all groups defined by the protected attribute :math:`A`.

  2. **Linear Programming Formulation**:
     
     The problem is expressed as a linear program in four variables. The coefficients of the linear program can be computed from the joint distribution of :math:`(\hat{Y}, A, Y)`. The objective function is a linear function of the parameters that specify :math:`\tilde{Y}`.

  3. **Solving the optimization problem**:

     Solve the linear program to find the optimal values for the parameters that define the derived predictor :math:`\tilde{Y}`. The solution will provide the predictor that satisfies the fairness constraint.

Basic Usage
~~~~~~~~~~~~~~

You can find an example of using the Linear Programming Debiaser method in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/binary_classification/demos/postprocessing.html#3.-LP-Debiaser>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.LPDebiaserBinary`.

References
~~~~~~~~~~~~~~
1. Hardt, Moritz, Eric Price, and Nati Srebro. "Equality of opportunity in supervised learning." Advances in neural information processing systems 29 (2016).
