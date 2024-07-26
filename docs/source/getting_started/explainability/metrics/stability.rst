.. default-role:: math

Stability Metrics
=================

The Data Stability and Feature Stability metrics are designed to evaluate the consistency of feature importance across different instances and features in a dataset. These metrics help quantify the robustness and reliability of feature importance, facilitating better model explainability and transparency.

.. contents:: Table of Contents
   :local:
   :depth: 1

Data Stability
----------------------

Methodology
~~~~~~~~~~~
The **Data Stability** metric evaluates the consistency of local feature importances across different instances. It measures how much the importances of features vary for different instances in a dataset.

Mathematical Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let :math:`\mathbf{I} = \{I_1, I_2, \ldots, I_n\}` be the set of local feature importances for \( n \) instances, where each \( I_i \) is a vector of feature importances.

1. **Calculation of Spread Divergence:**

.. math::

   S_i = \text{spread_divergence}(I_i)

2. **Calculation of Data Stability:**

.. math::

   \text{Data_Stability} = \text{spread_divergence}(\{S_i \mid i = 1, \ldots, n\})

Interpretation
~~~~~~~~~~~~~~~
- **High value:** Indicates that the feature importances are consistent across instances. This suggests that the model has a uniform understanding of the data, facilitating interpretation and increasing confidence in the model's explanations.
- **Low value:** Indicates that the feature importances vary significantly between instances. This can make the model harder to interpret and reduce confidence in its predictions.

The **Data Stability** metric uses spread divergence to evaluate the stability of feature importances. This divergence measures the dispersion of importances across different instances, providing a quantitative measure of consistency.


Feature Stability
-----------------

Methodology
~~~~~~~~~~~~
The **Feature Stability** metric measures the stability of individual feature importances across different instances. It focuses on the consistency of the importance of a specific feature throughout the dataset.

Mathematical Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let :math:`\mathbf{I} = \{I_1, I_2, \ldots, I_n\}` be the set of local feature importances for \( n \) instances, where each \( I_i \) is a vector of feature importances.

1. **Normalization and Transposition of Data:**

.. math::

   \mathbf{I}^T = \begin{pmatrix}
   I_{1,1} & I_{1,2} & \cdots & I_{1,n} \\
   I_{2,1} & I_{2,2} & \cdots & I_{2,n} \\
   \vdots & \vdots & \ddots & \vdots \\
   I_{m,1} & I_{m,2} & \cdots & I_{m,n}
   \end{pmatrix}

2. **Calculation of Spread Divergence for Each Feature:**

.. math::

   S_j = \text{spread_divergence}(I_j^T)

3. **Calculation of Feature Stability:**

.. math::

   \text{Feature_Stability} = \text{spread_divergence}(\{S_j \mid j = 1, \ldots, m\})

Interpretation
~~~~~~~~~~~~~~~
- **High value:** Indicates that the importance of a specific feature is consistent across instances. This suggests that the feature is robust and its relationship with the model's target is reliable.
- **Low value:** Indicates that the importance of a feature varies significantly between instances. This may suggest that the feature is less reliable and its relationship with the model's target may be weak.

The **Feature Stability** metric uses spread divergence to evaluate the stability of individual feature importances. This divergence measures the dispersion of the importances of each feature across different instances, providing a quantitative measure of their consistency.
