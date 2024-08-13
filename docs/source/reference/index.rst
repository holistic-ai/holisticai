.. _api:

=============
API Reference
=============

This page provides an overview of all public objects, functions, and methods in the ``holisticai`` package. The ``holisticai.*`` namespace includes all publicly exposed classes and functions.

Below is a list of key modules in ``holisticai``:

Bias
----

Bias in machine learning refers to unfair discrimination based on characteristics such as race, gender, age, or socioeconomic status. Addressing bias is crucial for ensuring fairness, transparency, and accountability in AI systems.

.. grid:: 3 3 3 3
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Metrics
        :link: bias/metrics.html
        :shadow: md

        Metrics for evaluating bias in classification, regression, clustering, and recommender systems.

    .. grid-item-card:: Mitigation
        :link: bias/mitigation.html
        :shadow: md

        Strategies to enhance fairness across various learning tasks.

    .. grid-item-card:: Plots
        :link: bias/plots.html
        :shadow: md

        Tools for visualizing bias in different learning tasks.

.. toctree::
   :hidden:
   :maxdepth: 3

   bias/metrics
   bias/mitigation
   bias/plots


Explainability
--------------

Explainability in machine learning is the ability to understand and interpret the decisions and predictions made by AI models. It is essential for ensuring transparency, trust, and accountability in AI systems.

.. grid:: 3 3 3 3
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Metrics
        :link: explainability/metrics.html
        :shadow: md

        Metrics for assessing explainability in classification, regression, clustering, and recommender systems.

    .. grid-item-card:: Plots
        :link: explainability/plots.html
        :shadow: md

        Visualization tools for explaining AI decisions across various learning tasks.

.. toctree::
   :hidden:

   explainability/metrics
   explainability/plots


Security
--------

Security in machine learning involves practices and measures to protect AI models and systems from malicious attacks, unauthorized access, and vulnerabilities. Ensuring security is vital for maintaining the integrity, confidentiality, and availability of AI systems, thereby fostering trust and reliability.

.. grid:: 3 3 3 3
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Metrics
        :link: security/metrics.html
        :shadow: md

        Metrics for evaluating security in classification and regression tasks.

    .. grid-item-card:: Mitigation
        :link: security/mitigation.html
        :shadow: md

        Strategies for enhancing security and robustness in learning tasks.

.. toctree::
   :hidden:

   security/metrics
   security/mitigation


Tools
-----

The following tools are available to support the construction and deployment of machine learning solutions:

.. grid:: 3 3 3 3
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Robustness
        :link: robustness/index.html
        :shadow: md

        Metrics and attackers for evaluating the robustness of machine learning models.

    .. grid-item-card:: Pipeline
        :link: pipeline.html
        :shadow: md

        A framework for building and deploying machine learning pipelines.

    .. grid-item-card:: Datasets
        :link: datasets.html
        :shadow: md

        Datasets for experimentation and testing.

.. If you update this toctree, also update the manual toctree in the
.. main index.rst.template

.. toctree::
   :hidden:

   pipeline
   datasets
   robustness