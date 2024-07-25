Matrix Factorization with Propensity Scoring
---------------------------------------------

.. note::
    **Learning tasks:** Recommendation systems.

Introduction
~~~~~~~~~~~~
Matrix Factorization with Propensity Scoring is a method designed to address selection biases in the evaluation and training of recommender systems. Selection biases often arise because users tend to rate items they like and ignore those they do not, leading to data that is Missing Not At Random (MNAR). This method adapts models and estimation techniques from causal inference to provide unbiased performance estimators and improve prediction performance. By viewing recommendations as interventions, similar to treatments in medical studies, this approach ensures accurate estimation of effects despite biased data.

Description
~~~~~~~~~~~

- **Problem definition**

  The primary problem addressed by this method is the selection bias inherent in the data used to train and evaluate recommender systems. Traditional matrix factorization methods do not account for the fact that observed ratings are not a random sample of all possible ratings. Instead, they are biased towards items that users are more likely to rate, typically those they like. This bias can lead to inaccurate performance evaluation and suboptimal recommendation models.

- **Main features**

  The main features of the Matrix Factorization with Propensity Scoring method include:
  
  1. **Unbiased Performance Estimation**: The method uses propensity-weighting techniques to derive unbiased estimators for various performance measures such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and Discounted Cumulative Gain (DCG).
  
  2. **Empirical Risk Minimization (ERM) Framework**: The method incorporates an ERM framework for learning recommendation systems under selection bias, providing generalization error bounds.
  
  3. **Propensity-Weighted Matrix Factorization**: The method extends traditional matrix factorization by incorporating propensity scores to account for selection bias, leading to improved prediction performance.
  
  4. **Robustness to Mis-specified Propensities**: The method is designed to be robust against inaccuracies in the estimated propensities, ensuring reliable performance even when the propensity model is not perfectly specified.

- **Step-by-step description of the approach**

  1. **Estimating Propensities**: 
     Propensities are the probabilities that a particular rating is observed. These are estimated using models such as Naive Bayes or logistic regression, based on user and item covariates. For example, in a movie recommendation system, propensities might be estimated based on user demographics and movie genres.

  2. **Unbiased Performance Estimation**:
     Using the estimated propensities, the method applies propensity-weighting to derive unbiased estimators for performance measures.

  3. **Empirical Risk Minimization (ERM)**:
     The method formulates the learning problem as an ERM problem, where the objective is to minimize the expected loss over the observed data, weighted by the inverse of the propensity scores. This ensures that the learned model is unbiased with respect to the true distribution of ratings.

  4. **Propensity-Weighted Matrix Factorization**:
     The matrix factorization model is modified to incorporate propensity scores.

  5. **Evaluation and Learning**:
     The method evaluates the performance of the recommender system using the unbiased estimators and iteratively updates the model parameters to minimize the propensity-weighted loss. This process continues until convergence, resulting in a model that is robust to selection biases.

References
~~~~~~~~~~~~~~~~
1. Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and Thorsten Joachims. 2016. Recommendations as treatments: Debiasing learning and evaluation. arXiv preprint arXiv:1602.05352 (2016).