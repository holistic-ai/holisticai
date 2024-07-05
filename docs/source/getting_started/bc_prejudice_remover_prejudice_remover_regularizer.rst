**Prejudice Remover Regularizer**
=================

**Introduction**
----------------
The Prejudice Remover Regularizer is a method designed to enforce fairness in classification models by reducing indirect prejudice. This method is particularly relevant in scenarios where sensitive features (e.g., race, gender) should not influence the prediction outcomes. The regularizer is integrated into any prediction algorithm models to ensure that the predictions are fair and unbiased.

**Description**
---------------
The method addresses a classification problem where the target variable :math:`Y` is binary (\{0,1\}), the non-sensitive features :math:`X` are real vectors, and the sensitive feature :math:`S` takes discrete values. The goal is to estimate the model parameters :math:`\Theta` such that the predictions are fair with respect to the sensitive feature :math:`S`.

The logistic regression model used is defined as:

.. math::
    :label: logistic-regression

    M[y|x,s; \Theta] = y \sigma(x^\top w_s) + (1 - y)(1 - \sigma(x^\top w_s))

where :math:`\sigma(\cdot)` is the sigmoid function, and :math:`\Theta = \{w_s\}_{s \in S}` are the weight vectors for :math:`x`.

The objective function to be minimized includes the log-likelihood, a standard L2 regularizer to avoid overfitting, and the prejudice remover regularizer :math:`R_{PR}`:

.. math::
    :label: objective-function

    -L(D; \Theta) + \eta R_{PR}(D, \Theta) + \frac{\lambda}{2} \|\Theta\|^2_2

where :math:`\lambda` and :math:`\eta` are positive regularization parameters.

The prejudice remover regularizer :math:`R_{PR}` is designed to reduce the indirect prejudice by minimizing the prejudice index (PI), which is defined as the mutual information between the target variable :math:`Y` and the sensitive feature :math:`S`:

.. math::
    :label: prejudice-index

    PI = \sum_{(y,s) \in D} \hat{Pr}[y,s] \ln \frac{\hat{Pr}[y,s]}{\hat{Pr}[y] \hat{Pr}[s]}

The regularizer :math:`R_{PR}` is then formulated as:

.. math::
    :label: regularizer

    R_{PR}(D, \Theta) = \sum_{(x_i, s_i) \in D} \sum_{y \in \{0,1\}} M[y|x_i, s_i; \Theta] \ln \frac{\hat{Pr}[y|s_i]}{\hat{Pr}[y]}

**Equations/Algorithms**
------------------------

The logistic regression model:

.. math::
    :label: logistic-regression

    M[y|x,s; \Theta] = y \sigma(x^\top w_s) + (1 - y)(1 - \sigma(x^\top w_s))

The objective function to be minimized:

.. math::
    :label: objective-function

    -L(D; \Theta) + \eta R_{PR}(D, \Theta) + \frac{\lambda}{2} \|\Theta\|^2_2

The prejudice index (PI):

.. math::
    :label: prejudice-index

    PI = \sum_{(y,s) \in D} \hat{Pr}[y,s] \ln \frac{\hat{Pr}[y,s]}{\hat{Pr}[y] \hat{Pr}[s]}

The prejudice remover regularizer :math:`R_{PR}`:

.. math::
    :label: regularizer

    R_{PR}(D, \Theta) = \sum_{(x_i, s_i) \in D} \sum_{y \in \{0,1\}} M[y|x_i, s_i; \Theta] \ln \frac{\hat{Pr}[y|s_i]}{\hat{Pr}[y]}

**Usage Examples**
------------------
The Prejudice Remover Regularizer was tested on various datasets to evaluate its effectiveness in reducing indirect prejudice while maintaining prediction accuracy. For instance, it was applied to the Census dataset where the sensitive feature was gender. The method successfully reduced the prejudice index (PI) while achieving a reasonable trade-off with prediction accuracy.

**Advantages and Limitations**
------------------------------

*Advantages:*

- Effectively reduces indirect prejudice in classification models.
- Can be integrated into any prediction algorithm frameworks with modest computational resources.
- Provides a quantifiable measure of fairness through the prejudice index (PI).

*Limitations:*

- May sacrifice prediction accuracy to achieve fairness, depending on the regularization parameter :math:`\eta`.
- The method's performance is sensitive to the choice of regularization parameters :math:`\lambda` and :math:`\eta`.
- Requires careful tuning to balance the trade-off between fairness and accuracy.

**References**
---------------
1. Kamishima, Toshihiro, et al. "Fairness-aware classifier with prejudice remover regularizer." Joint European conference on machine learning and knowledge discovery in databases. Springer, Berlin, Heidelberg, 2012.