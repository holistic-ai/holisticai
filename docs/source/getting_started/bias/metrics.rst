
Metrics
=======

Bias is defined as an unwanted prejudice in the decisions made by an AI system that systematically disadvantages a person or group. Various types of bias can exist and may be unintentionally introduced into algorithms at any stage of the development process, whether during data generation or model building.

    In order to measure whether a system treats different groups of people equally, we can follow two approaches: **equality of outcome** and **equality of opportunity**.

When we select **equality of outcome**, we ask that all subgroups have equal outcomes. For example, in a recruitment context, we may require that the percentage of applicants hired is consistent across groups (e.g. we want to hire 5% of all female applicants and 5% of all male applicants). Mathematically, this means that the likelihood of a positive outcome is equal for members of each group (regardless of the ground-truth labels):

.. math::
    P(\hat{Y} = 1 | G = a) = P(\hat{Y} = 1 | G = b) \quad \forall a, b

that is, the probability of a positive outcome is the same for all groups. 

When we select **equality of opportunity**, we ask that all subgroups are given the same opportunity of outcomes. For example, if we have a face recognition algorithm, we may want the classifier to perform equally well for all ethnicities and genders. Mathematically, the probability of a person in the positive class being correctly assigned a positive outcome and the probability of a person in a negative class being incorrectly assigned a positive outcome should both be the same for privileged and unprivileged group members. In this case, ground-truth labels are used to define the groups, and the following condition should hold:

.. math::
    P(\hat{Y} = 1 | Y = y, G = a) = P(\hat{Y} = 1 | Y = y, G = b) \quad \forall a, b ~~and~~ y \in \{0, 1\}

that is, the probability of a positive outcome is the same for all groups, given the ground-truth labels.


.. toctree::
    :maxdepth: 1

    metrics/binary_classification
    metrics/multi_classification
    metrics/regression
    metrics/recommender
    metrics/clustering

Summary Table
-------------

The following table summarizes the metrics that can be used to measure bias in different types of tasks.

.. csv-table:: Bias Metrics
    :header: "Class", "Task", "Metrics", "Ideal Value", "Fair Area", "Description"
    :file: bias_metrics.csv
