Spread Metrics
==============

.. contents:: Table of Contents
   :local:
   :depth: 1

Alpha Importance Score
----------------------

The Alpha Importance Score (AIS) metric evaluates the proportion of features that have an importance value greater than or equal to a specified threshold (alpha). This metric helps in understanding the concentration of importance among the most significant features, providing insights into model interpretability and feature relevance.

Methodology
~~~~~~~~~~~

1. **Filter Feature Importance:**
   - Identify features with importance values greater than or equal to a specified threshold, alpha.

2. **Calculate Alpha Importance:**
   - Compute the ratio of the number of features with importance values greater than or equal to alpha to the total number of features.

Mathematical Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let \( I \) denote the list of feature importance values, and \( \alpha \) be the specified threshold. The filtered feature importance list \( I_\alpha \) is given by:

.. math::

   I_\alpha = \{ i \in I \mid i \geq \alpha \}

The Alpha Importance Score \( AIS \) is then calculated as:

.. math::

   AIS = \frac{| I_\alpha |}{|I|}

where :math:`|{I_\alpha}|` is the number of features with importance greater than or equal to :math:`\alpha`, and :math:`|I|` is the total number of features.

Interpretation
~~~~~~~~~~~~~~

- **High score:** Indicates that a significant proportion of features have high importance values, suggesting that the model relies heavily on a few key features.
- **Low score:** Indicates that fewer features have high importance values, suggesting a more even distribution of feature importance.




Feature Spread Metric
----------------------


Methodology
~~~~~~~~~~~~~~

1. **Normalize Feature Importance**:
   - For a given set of features \( F = [F_1, F_2, \ldots, F_n] \), normalize the feature importance using a normalization function \( N \) such that the sum of the normalized feature importance values equals 1.
   - Mathematically, for each feature \( F_i \), the normalized feature importance \( P(F_i) \) is given by:
   
     .. math::

        P(F_i) = \frac{F_i}{\sum_{j=1}^{n} F_j}

   
2. **Calculate Jensen–Shannon Divergence**:
   - Jensen–Shannon divergence (JSD) measures the similarity between the normalized feature importance distribution and a uniform distribution.
   - It is calculated using the formula:

     .. math::

        JSD(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)


   - where \( D_{KL} \) is the Kullback-Leibler divergence, \( P \) is the normalized feature importance distribution, \( Q \) is the uniform distribution, and \( M = \frac{1}{2}(P + Q) \).

Mathematical Representation
~~~~~~~~~~~~~~

Let \( P \) denote the normalized feature importance vector and \( Q \) denote the uniform distribution vector. The Jensen–Shannon Divergence \( JSD \) is given by:

.. math::

   JSD(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)

where:

.. math::

   M = \frac{1}{2}(P + Q)

and

.. math::

   D_{KL}(P \| M) = \sum_{i=1}^{n} P(F_i) \log \left( \frac{P(F_i)}{M_i} \right)

Interpretation
~~~~~~~~~~~~~~

We use the inverse of the Jensen–Shannon Divergence as the Spread Divergence Metric.

- **High Score**: Indicates that the distribution of feature importance is close to uniform, suggesting that the model relies on a broader set of features. This implies lower interpretability.
- **Low Score**: Indicates that the distribution of feature importance is far from uniform, suggesting that the model relies on fewer, more significant features. This implies higher interpretability.