Feature Permutation-Based Metric
========================

.. contents:: Table of Contents
   :local:
   :depth: 1

Explainability Ease Score Metric
--------------

The Explainability Ease Score (E) metric evaluates the simplicity of interpreting the partial dependence curve of a feature. This metric quantifies how easily these curves can be interpreted, facilitating better model explainability and transparency.

Methodology
~~~~~~~~~~~

To compute the Explainability Ease Score, follow these steps:

1. **Compute the Partial Dependence Curve:**
   - Calculate the partial dependence of the feature of interest. This curve shows the average effect of the feature on the predicted outcome.

2. **Compute the Second Derivative:**
   - Compute the numerical second derivative of the partial dependence curve. The second derivative represents the rate of change of the slope of the curve.
   - If the curve is linear, the second derivative will be zero. Non-linear curves will have a non-zero second derivative.

3. **Normalize the Second Derivative:**
   - Normalize the second derivative vector by computing the norm of the vector containing the absolute values of the second derivative at multiple points along the feature's domain.

4. **Compare Tangents:**
   - Divide the partial dependence curve into three sections and compute the average slope for each section.
   - Calculate the cosine similarity between the slopes of consecutive sections.

5. **Compute Feature Scores:**
   - Based on the cosine similarity of the slopes, assign scores to each feature. A feature with more consistent slopes (higher cosine similarity) indicates easier interpretability.

6. **Calculate the Explainability Ease Score:**
   - The Explainability Ease Score is computed as the weighted average of the feature scores, normalized to a range from 0 to 1.
   - A higher score (close to 1) indicates easier interpretability, while a lower score (close to 0) indicates more complex interpretability.

Mathematical Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let \( f_{x''} \) denote the second derivative of the partial dependence curve for feature \( x \), computed over \( q \) points in the domain of \( x \). The Explainability Ease Score is given by:

.. math::

   E_{x} := \| \vec{v_{x}} \|, \quad \text{where} \quad \vec{v_{x}} := \left\{ |f_{x''}(x_{1})|, |f_{x''}(x_{2})|, \ldots, |f_{x''}(x_{q})| \right\}

The score for each feature is then computed based on the similarity of the slopes in different sections of the partial dependence curve. Let \( S_i \) be the slope of section \( i \), and \( \cos(S_i, S_{i+1}) \) be the cosine similarity between consecutive sections. The feature score is computed as:

.. math::

   \text{Feature Score} = \sum_{i=1}^{n-1} \cos(S_i, S_{i+1})

The final Explainability Ease Score \( E \) is the normalized weighted average of the feature scores:

.. math::

   E = \frac{1}{n} \sum_{j=1}^{n} \text{Feature Score}_j

where \( n \) is the number of features. 

Interpretation
~~~~~~~~~~~~~~

The score \( E \) ranges from 0 to 1, where:

- **High Score :** Indicates a simple, linear relationship between the feature and the predicted outcome. These curves are easy to interpret.
- **Low EScore :** Indicates a complex, non-linear relationship. These curves are harder to interpret and may require more sophisticated analysis to understand.sis to understand.