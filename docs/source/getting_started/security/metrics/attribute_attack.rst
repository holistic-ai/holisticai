Attribute Inference Attack
==========================

This section details the implementation of a simple black-box attribute inference attack, a technique used to infer sensitive attributes from a dataset based on the non-sensitive attributes and the model's predictions.

Overview
--------

The attribute inference attack involves training an estimator to predict the sensitive attribute (the attacked feature) using the other features and the true class labels during training. During the attack phase, it uses the non-sensitive features and the model's predictions. This method assumes that the attacker has access to the target model's predictions for the samples under attack.

Methodology
-----------

1. **Data Collection**:
   - Gather the dataset including both sensitive (attacked) and non-sensitive features.
   - Ensure the dataset contains true class labels for training purposes.

2. **Feature Preparation**:
   - Separate the dataset into non-sensitive features (input features) and the sensitive feature (target feature).
   - Remove the sensitive feature from the dataset.
   - During training, combine the non-sensitive features with the true class labels.

3. **Model Training**:
   - Train an estimator where the input is the combination of non-sensitive features and true class labels, and the output is the sensitive feature.
   - Use standard training techniques and appropriate metrics to evaluate the performance of the estimator.

4. **Inference (Attack Phase)**:
   - Use the trained estimator to predict the sensitive attribute for new samples based on their non-sensitive features and the model’s predictions.

Example Workflow
-----------------

1. **Load Data**:
   - Load the dataset containing both sensitive and non-sensitive features.

2. **Preprocess Data**:
   - Separate the sensitive feature from the dataset.
   - Combine the non-sensitive features with the true class labels for training.

3. **Train Attribute Inference Model**:
   - Use the non-sensitive features and true class labels as input to train the estimator.
   - Train the estimator to predict the sensitive feature.

4. **Attack Phase**:
   - Obtain predictions from the target model for each sample in the dataset.
   - Use the non-sensitive features and the model’s predictions as input to the trained estimator to infer the sensitive attribute.

Accuracy Calculation
---------------------

Classification
~~~~~~~~~~~~~~

To evaluate the performance of the attribute inference attack, the accuracy of the estimator is calculated using the following equation:

.. math::

    \text{Accuracy} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}\{\hat{y}_i = y_i\}

Where:

- :math:`n` is the total number of samples.
- :math:`\hat{y}_i` is the predicted value of the sensitive attribute for the :math:`i`-th sample.
- :math:`y_i` is the true value of the sensitive attribute for the :math:`i`-th sample.
- :math:`\mathbf{1}\{\cdot\}` is the indicator function that returns 1 if the condition inside is true, and 0 otherwise.

Regression
~~~~~~~~~~

For regression tasks, the Mean Squared Error (MSE) is commonly used to measure the performance of the attribute inference attack. The MSE is calculated as follows:

.. math::

    \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2

Where:

- :math:`n` is the total number of samples.
- :math:`y_i` is the true value of the sensitive attribute for the :math:`i`-th sample.
- :math:`\hat{y}_i` is the predicted value of the sensitive attribute for the :math:`i`-th sample.

The MSE provides an indication of how close the predicted values are to the true values, with lower values indicating better performance.
