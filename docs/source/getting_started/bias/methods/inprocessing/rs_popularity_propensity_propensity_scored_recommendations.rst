Propensity-Scored Recommendations
---------------------

.. note::
    **Learning tasks:** Recommendation systems.

Introduction
~~~~~~~~~~~~~~~~
Propensity-scored recommendations are a method designed to handle selection biases in the evaluation and training of recommender systems. This approach leverages techniques from causal inference to provide unbiased performance estimators and improve prediction accuracy, even when the data is Missing Not At Random (MNAR).

Description
~~~~~~~~~~~~~~~~
The propensity-scored recommendations method addresses the problem of biased data in recommender systems, where users typically rate items they like, leading to a non-random missing data pattern. The method involves estimating the probability (propensity) that a user will rate an item and using these propensities to adjust the training and evaluation processes.

The main characteristics of the method include:

- Estimating propensities for each user-item pair.
- Using these propensities to weight the observed data.
- Applying the Inverse-Propensity-Scoring (IPS) estimator to obtain unbiased performance measures.
- Integrating propensity scores into an Empirical Risk Minimization (ERM) framework for learning.

References
~~~~~~~~~~~~~~~~
1. Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and Thorsten Joachims. 2016. Recommendations as treatments: Debiasing learning and evaluation. arXiv preprint arXiv:1602.05352 (2016).