==========
Mitigation
==========

.. warning::

   This page is currently under construction. Please check back later for updates.

.. contents:: **Contents:**
    :depth: 2

Introduction
------------

   This page is currently under construction. Please check back later for updates.

.. note::
    Bias mitigation can be approached through **pre-processing**, **in-processing**, and **post-processing** methods.

Bias in AI systems can lead to systematic disadvantages for certain individuals or groups. To address and reduce these biases, several mitigation methods can be employed throughout the development process, from data generation to model deployment. This document outlines various techniques and strategies for mitigating bias in AI systems.

**Pre-processing Methods**

Pre-processing techniques aim to mitigate bias by transforming the data before it is used to train a model. This can include:

- **Reweighting**: Adjusting the weights of data points to ensure balanced representation across groups.
- **Resampling**: Over-sampling underrepresented groups or under-sampling overrepresented groups to achieve a balanced dataset.
- **Data Augmentation**: Generating synthetic data to bolster the representation of minority groups.
- **Fair Representation Learning**: Learning new representations of data that remove sensitive information while retaining useful information for prediction.

**In-processing Methods**

In-processing techniques modify the learning algorithm itself to reduce bias during the model training phase. This can include:

- **Adversarial Debiasing**: Training the model in conjunction with an adversary that penalizes the model for producing biased predictions.
- **Fairness Constraints**: Incorporating constraints that enforce fairness criteria directly into the optimization objective of the learning algorithm.
- **Regularization Techniques**: Adding regularization terms to the loss function that penalize bias.

**Post-processing Methods**

Post-processing techniques adjust the predictions of the trained model to mitigate bias. This can include:

- **Calibrated Equalized Odds**: Adjusting the decision thresholds for different groups to achieve equalized odds across groups.
- **Reject Option Classification**: Allowing uncertain predictions to be reconsidered, particularly for disadvantaged groups, to improve fairness.
- **Output Adjustment**: Modifying the output probabilities to ensure fair treatment across groups.

.. toctree::
    :maxdepth: 1

    mitigation/preprocessing
    mitigation/inprocessing
    mitigation/postprocessing

Summary Table
-------------

The following table summarizes the bias mitigation methods and their applicability at different Machine Learning (ML) tasks.

.. csv-table:: **Bias Mitigation Methods**
    :header: "Type Strategy", "Method name", "Binary Classification", "Multi Classification", "Regression", "Clustering", "Recommender Systems"
    :file: bias_mitigation.csv