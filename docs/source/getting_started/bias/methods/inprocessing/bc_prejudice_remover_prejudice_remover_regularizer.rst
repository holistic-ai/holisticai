Prejudice Remover Regularizer
-----------------------------

.. note::
    **Learning tasks:** Binary classification.

Introduction
~~~~~~~~~~~~
The Prejudice Remover Regularizer is a method designed to enforce fairness in classification tasks by reducing indirect prejudice. This method is integrated into logistic regression models and aims to ensure that the predictions are not unfairly influenced by sensitive features. The regularizer is designed to be computationally efficient and easy to implement, making it a practical solution for fairness-aware machine learning.

Description
~~~~~~~~~~~

- **Problem definition**

  In classification tasks, we often deal with a target variable :math:`Y`, non-sensitive features :math:`X`, and a sensitive feature :math:`S`. The goal is to predict the class :math:`Y` based on :math:`X` and :math:`S` while ensuring that the sensitive feature :math:`S` does not unfairly influence the prediction. The training dataset is denoted as :math:`D = \{(y, x, s)\}`, where :math:`y`, :math:`x`, and :math:`s` are instances of :math:`Y`, :math:`X`, and :math:`S`, respectively.

- **Main features**

  The method incorporates two types of regularizers into the logistic regression model:
  
  1. **L2 Regularizer**: This is a standard regularizer used to avoid overfitting. It is represented as :math:`\|\Theta\|_2^2`, where :math:`\Theta` is the set of model parameters.
  
  2. **Prejudice Remover Regularizer**: This regularizer, denoted as :math:`R_{PR}`, is specifically designed to reduce indirect prejudice by minimizing the prejudice index (PI). The PI quantifies the statistical dependence between the sensitive feature :math:`S` and the target variable :math:`Y`.

- **Step-by-step description of the approach**

  1. **Model Definition**: The conditional probability of a class given non-sensitive and sensitive features is modeled by :math:`M[Y|X,S;\Theta]`, where :math:`\Theta` is the set of model parameters. The logistic regression model is used as the prediction model:
     
     .. math::
        M[y|x,s;\Theta] = y\sigma(x^\top w_s) + (1-y)(1-\sigma(x^\top w_s)),
     
     where :math:`\sigma(\cdot)` is the sigmoid function, and :math:`w_s` are the weight vectors for :math:`x`.

  2. **Log-Likelihood Maximization**: The parameters are estimated based on the maximum likelihood principle, aiming to maximize the log-likelihood:
     
     .. math::
        L(D;\Theta) = \sum_{(y_i, x_i, s_i) \in D} \ln M[y_i|x_i, s_i; \Theta].

  3. **Objective Function**: The objective function to minimize is obtained by adding the L2 regularizer and the prejudice remover regularizer to the negative log-likelihood:
     
     .. math::
        -L(D;\Theta) + \eta R(D, \Theta) + \frac{\lambda}{2} \|\Theta\|_2^2,
     
     where :math:`\lambda` and :math:`\eta` are positive regularization parameters.

  4. **Prejudice Index Calculation**: The prejudice index (PI) is defined as:
     
     .. math::
        PI = \sum_{Y,S} \hat{Pr}[Y,S] \ln \frac{\hat{Pr}[Y,S]}{\hat{Pr}[S] \hat{Pr}[Y]}.
     
     This can be approximated using the training data:
     
     .. math::
        PI \approx \sum_{(x_i, s_i) \in D} \sum_{y \in \{0,1\}} M[y|x_i, s_i; \Theta] \ln \frac{\hat{Pr}[y|s_i]}{\hat{Pr}[y]}.

  5. **Normalization**: The normalized prejudice index (NPI) is calculated to quantify the degree of indirect prejudice:
     
     .. math::
        NPI = \frac{PI}{\sqrt{H(Y)H(S)}},
     
     where :math:`H(\cdot)` is the entropy function.

  6. **Trade-off Efficiency**: The efficiency of the trade-off between prediction accuracy and prejudice removal is measured by the ratio :math:`PI/MI`, where :math:`MI` is the mutual information between the predicted and true labels.


References
~~~~~~~~~~~~~~~~
1. Kamishima, Toshihiro, et al. "Fairness-aware classifier with prejudice remover regularizer." Joint European conference on machine learning and knowledge discovery in databases. Springer, Berlin, Heidelberg, 2012.