Stability Metrics
=================

The Data Stability and Feature Stability metrics are designed to evaluate the consistency of feature importance across different instances and features in a dataset. These metrics help quantify the robustness and reliability of feature importance, facilitating better model explainability and transparency.

.. contents:: Table of Contents
   :local:
   :depth: 1

Data Stability
--------------

Data Stability measures the consistency of feature importance across different instances in the dataset. A high data stability score indicates that the importance of features remains relatively stable across various data points, enhancing interpretability and reliability.

Methodology
~~~~~~~~~~~

1. **Compute Feature Importance for Each Instance:**
   - Calculate the feature importance for each instance in the dataset.

2. **Calculate Median and Interquartile Range (IQR) for Each Instance:**
   - Compute the median feature importance for each instance.
   - Compute the interquartile range (IQR) for each instance, defined as the difference between the 75th percentile (Q3) and the 25th percentile (Q1) of feature importance values.

3. **Calculate Normalized Interquartile Range (nIQR) for Each Instance:**
   - Normalize the IQR by dividing it by the median feature importance for each instance.

4. **Compute Data Stability Score:**
   - Aggregate the normalized IQR values across all instances to compute the Data Stability Score. A lower normalized IQR indicates higher stability.

Mathematical Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let \( F_i \) denote the feature importance vector for instance \( i \). The median and IQR for instance \( i \) are given by:

.. math::

   \text{Median}_i = \text{median}(F_i)

   \text{IQR}_i = Q3(F_i) - Q1(F_i)

The normalized IQR (nIQR) for instance \( i \) is:

.. math::

   \text{nIQR}_i = \frac{\text{IQR}_i}{\text{Median}_i}

The Data Stability Score \( \lambda_D \) is the mean of the nIQR values:

.. math::

   \lambda_D = \frac{1}{N} \sum_{i=1}^{N} \text{nIQR}_i

where \( N \) is the number of instances.

Interpretation
~~~~~~~~~~~~~~

- **High Score:** Indicates that the importance of features is consistent across different instances, making the model more interpretable and reliable.
- **Low Score:** Indicates that the importance of features varies significantly across different instances, making the model less interpretable and potentially less reliable.


Feature Stability
-----------------

Feature Stability measures the consistency of feature importance across different features in the dataset. A high feature stability score indicates that the importance of features remains relatively stable across various data points, enhancing interpretability and reliability.

Methodology
~~~~~~~~~~~

1. **Compute Feature Importance for Each Feature:**
   - Calculate the feature importance for each feature across all instances in the dataset.

2. **Calculate Median and Interquartile Range (IQR) for Each Feature:**
   - Compute the median feature importance for each feature.
   - Compute the interquartile range (IQR) for each feature, defined as the difference between the 75th percentile (Q3) and the 25th percentile (Q1) of feature importance values.

3. **Calculate Normalized Interquartile Range (nIQR) for Each Feature:**
   - Normalize the IQR by dividing it by the median feature importance for each feature.

4. **Compute Feature Stability Score:**
   - Aggregate the normalized IQR values across all features to compute the Feature Stability Score. A lower normalized IQR indicates higher stability.

Mathematical Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let \( F_j \) denote the feature importance vector for feature \( j \). The median and IQR for feature \( j \) are given by:

.. math::

   \text{Median}_j = \text{median}(F_j)

   \text{IQR}_j = Q3(F_j) - Q1(F_j)

The normalized IQR (nIQR) for feature \( j \) is:

.. math::

   \text{nIQR}_j = \frac{\text{IQR}_j}{\text{Median}_j}

The Feature Stability Score \( \lambda_F \) is the mean of the nIQR values:

.. math::

   \lambda_F = \frac{1}{M} \sum_{j=1}^{M} \text{nIQR}_j

where \( M \) is the number of features.

Interpretation
~~~~~~~~~~~~~~

- **High Score:** Indicates that the importance of features is consistent across different features, making the model more interpretable and reliable.
- **Low Score:** Indicates that the importance of features varies significantly across different features, making the model less interpretable and potentially less reliable.