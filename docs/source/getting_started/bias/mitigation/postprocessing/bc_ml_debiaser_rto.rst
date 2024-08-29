Randomized Threshold Optimizer (RTO)
-------------------------------------

.. note::
    **Learning tasks:** Binary classification, Regression.

Introduction
~~~~~~~~~~~~
The Randomized Threshold Optimizer (RTO) is a post-processing algorithm designed to ensure fairness in machine learning models by adjusting their decision thresholds. The method is particularly effective in scenarios where models are trained at scale, such as with large datasets and over-parameterized models. RTO aims to achieve statistical parity, a fairness criterion, by learning a randomized prediction rule that adjusts the decision thresholds based on the sensitive attribute of the instances.

Description
~~~~~~~~~~~

- **Problem definition**

  The primary goal of RTO is to mitigate bias in machine learning models by ensuring that the predictions are fair across different groups defined by a sensitive attribute (e.g., gender, race). The method focuses on achieving :math:`\epsilon`-statistical parity, which requires that the difference in the expected predictions between any two groups does not exceed a specified threshold :math:`\epsilon`. This is formalized as:

  .. math::
      \max_{k \in [K]} \mathbb{E}[h(x) | x \in X_k] - \min_{k \in [K]} \mathbb{E}[h(x) | x \in X_k] \leq \epsilon

  where :math:`h(x)` is the prediction rule, :math:`X_k` represents the instances belonging to group :math:`k`, and :math:`K` is the total number of groups.

- **Main features**

  The RTO method introduces several key features to achieve its objectives:
  
  1. **Randomized Prediction Rule**: The method learns a prediction rule :math:`h_\gamma(x)` that incorporates randomization near the decision thresholds. This randomization is controlled by a hyperparameter :math:`\gamma`.
  
  2. **Fairness Constraints**: The algorithm ensures that the learned prediction rule satisfies the :math:`\epsilon`-statistical parity constraint.
  
  3. **Optimization Framework**: RTO formulates the problem as a regularized optimization problem, which is solved using a projected stochastic gradient descent (SGD) method.
  
  4. **Theoretical Guarantees**: The method provides theoretical guarantees for fairness and Bayes risk consistency, ensuring that the randomization near the thresholds is necessary and sufficient for achieving the desired fairness.

- **Step-by-step description of the approach**

  1. **Initialization**: The algorithm starts by initializing the parameters :math:`(\lambda_1, \mu_1), \ldots, (\lambda_K, \mu_K)` to zeros. These parameters are optimization variables that help the decision thresholds for each group.

  2. **Sampling and Updates**: The algorithm iteratively samples an instance :math:`x` from the data distribution and updates the parameters based on the gradients of a specially designed loss function. 
  
  3. **Convergence**: The updates are repeated until convergence, ensuring that the parameters stabilize and the fairness constraints are satisfied.

  4. **Prediction**: For a given instance :math:`x` in group :math:`X_k`, the prediction rule :math:`h_\gamma(x)` is defined as:
  
     .. math::
         h_\gamma(x) = \left[ \min \left\{ 1, \frac{f(x) - \lambda_k + \mu_k}{\gamma} \right\} \right]^+
  
     This rule ensures that the predictions are adjusted based on the learned thresholds and the randomization parameter :math:`\gamma`.

Basic Usage
~~~~~~~~~~~~~~

You can find an example of using the ML Debiaser module in the following `demo <https://holisticai.readthedocs.io/en/latest/gallery/tutorials/bias/mitigating_bias/binary_classification/demos/postprocessing.html#4.-ML-Debiaser>`_.

Read more about the class attributes and methods in the API reference: :class:`~holisticai.bias.mitigation.MLDebiaser`.

References
~~~~~~~~~~~~~~
1. Alabdulmohsin, Ibrahim M., and Mario Lucic. "A near-optimal algorithm for debiasing trained machine learning models." Advances in Neural Information Processing Systems 34 (2021): 8072-8084.
