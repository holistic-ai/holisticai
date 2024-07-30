.. _robustness_binary_classification_api:

============================
Robustness in Binary Classification
============================

This page provides an overview of all public ``holisticai`` objects, functions, and methods related to robustness in binary classification. All classes and functions exposed in the ``holisticai.*`` namespace are public.

Below is a list of modules for metrics and attackers:

Metrics
=======

Metrics to evaluate the robustness and stability of binary classification models.

.. autosummary::
    :toctree: .generated/

    holisticai.robustness.metrics.adversarial_accuracy
    holisticai.robustness.metrics.empirical_robustness

Attackers
=========

Techniques and strategies to simulate adversarial attacks and test model robustness.

.. autosummary::
    :toctree: .generated/

    holisticai.robustness.attackers.BinClassAttacker