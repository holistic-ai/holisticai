.. default-role:: math

Stability Metrics
=================

Feature Stability metrics are designed to evaluate the consistency of feature importance across different instances and features in a dataset. These metrics help quantify the robustness and reliability of feature importance, facilitating better model explainability and transparency.

.. contents:: Table of Contents
   :local:
   :depth: 1

Feature Stability
-----------------

Methodology
~~~~~~~~~~~
The **Feature Stability** metric measures the stability of individual feature importances across different instances. It focuses on the consistency of the importance of a specific feature throughout the dataset.

Mathematical Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let :math:`\mathbf{I} = \{I_1, I_2, \ldots, I_n\}` be the set of local feature importances for :math:`n` instances, where each :math:`I_i` is a vector of feature importances.

1. **Normalization of Data:**
   Each vector :math:`I_i` is normalized so that the sum of its elements equals 1:

   .. math::

      I_{i,j} \leftarrow \frac{I_{i,j}}{\sum_{k=1}^{m} I_{i,k}} \quad \text{for } i = 1, 2, \ldots, n \text{ and } j = 1, 2, \ldots, m

   where :math:`m` is the number of features.

2. **Computation of Importance Distributions:**
   The importance distribution :math:`\mathbf{D}` of features is computed by finding the density distribution of feature importance vectors. This is done by evaluating the proximity of these vectors to a set of synthetic samples generated from a Dirichlet distribution:

   .. math::

      \mathbf{D} = \left( d_1, d_2, \ldots, d_{m} \right)

   where :math:`d_j` represents the density estimate for feature :math:`j`.

3. **Calculation of Feature Stability:**
   Feature Stability is computed using one of the following strategies:

   - **Variance Strategy:** 
     The stability is determined by the ratio of the standard deviation to the maximum density:

     .. math::

        \textrm{FS} = 1 - \frac{\sigma_D}{\max(D)}

     where :math:`\sigma_D` represents the standard deviation of the density distribution :math:`\mathbf{D}`.

   - **Entropy Strategy:**
     Alternatively, the stability can be computed based on the Jensen-Shannon divergence between the distribution :math:`\mathbf{D}` and a uniform distribution:

     .. math::

        \textrm{FS} = 1 - \text{JSD}\left(\mathbf{D} \| \mathbf{U} \right)

     where :math:`\mathbf{U}` is the uniform distribution, and :math:`\text{JSD}` denotes the Jensen-Shannon divergence.

Interpretation
~~~~~~~~~~~~~~
- **High value:** Indicates that the importance of a specific feature is consistent across instances. This suggests that the feature is robust and its relationship with the model's target is reliable.
- **Low value:** Indicates that the importance of a feature varies significantly between instances. This may suggest that the feature is less reliable and its relationship with the model's target may be weak.

The **Feature Stability** metric provides a quantitative measure of the consistency of feature importances across different instances by evaluating the dispersion of these importances using either variance-based or entropy-based methods.
