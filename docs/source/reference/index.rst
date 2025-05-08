=============
API Reference
=============

This page provides an overview of all public objects, functions, and methods in the ``holisticai`` package. The ``holisticai.*`` namespace includes all publicly exposed classes and functions.

Below is a list of key modules in ``holisticai``:

Tools
-----

The following tools support the construction and deployment of machine learning solutions:

.. list-table:: Tools
   :header-rows: 1

   * - Module
     - Description
   * - `Pipeline <pipeline.html>`_
     - A framework for building and deploying machine learning pipelines.
   * - `Datasets <datasets.html>`_
     - Datasets for experimentation and testing.

Bias
----

Bias in machine learning refers to unfair discrimination based on characteristics such as race, gender, age, or socioeconomic status. Addressing bias is crucial for ensuring fairness, transparency, and accountability in AI systems.

.. list-table:: Bias
   :header-rows: 1

   * - Module
     - Description
   * - `Metrics <bias/metrics.html>`_
     - Metrics for evaluating bias in classification, regression, clustering, and recommender systems.
   * - `Mitigation <bias/mitigation.html>`_
     - Strategies to enhance fairness across various learning tasks.
   * - `Plots <bias/plots.html>`_
     - Tools for visualizing bias in different learning tasks.

Explainability
--------------

Explainability in machine learning is the ability to understand and interpret the decisions and predictions made by AI models. It is essential for ensuring transparency, trust, and accountability in AI systems.

.. list-table:: Explainability
   :header-rows: 1

   * - Module
     - Description
   * - `Metrics <explainability/metrics.html>`_
     - Metrics for assessing explainability in classification, regression, clustering, and recommender systems.
   * - `Plots <explainability/plots.html>`_
     - Visualization tools for explaining AI decisions across various learning tasks.

Security
--------

Security in machine learning involves practices and measures to protect AI models and systems from malicious attacks, unauthorized access, and vulnerabilities. Ensuring security is vital for maintaining the integrity, confidentiality, and availability of AI systems, thereby fostering trust and reliability.

.. list-table:: Security
   :header-rows: 1

   * - Module
     - Description
   * - `Metrics <security/metrics.html>`_
     - Metrics for evaluating security in classification and regression tasks.
   * - `Mitigation <security/mitigation.html>`_
     - Strategies for enhancing security and robustness in learning tasks.
   * - `Attackers <security/attackers.html>`_
     - Techniques and strategies to simulate attacks and test model security.

Robustness
----------

Robustness in machine learning refers to the ability of AI models to perform well under various conditions, including noisy data, adversarial attacks, and distribution shifts. Enhancing robustness is essential for ensuring the reliability, generalization, and resilience of AI systems.

.. list-table:: Robustness
   :header-rows: 1

   * - Module
     - Description
   * - `Metrics <robustness/metrics.html>`_
     - Metrics to evaluate the stability and robustness of models in various learning tasks.
   * - `Attackers <robustness/attackers.html>`_
     - Techniques and strategies to simulate adversarial attacks and test model robustness.
