.. _robustness_api:

========
Robustness
========

Robustness in machine learning refers to the ability of AI models and systems to maintain performance and stability under a variety of conditions, including noisy data, adversarial attacks, and changes in the operational environment. Ensuring robustness is essential for the reliability and resilience of AI systems, allowing them to function effectively even when faced with unexpected challenges.

Below is a list of modules in ``holisticai`` for machine learning tasks:

.. grid:: 3 3 3 3
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Metrics
        :link: binary_classification/metrics.html
        :shadow: md

        Metrics to evaluate the stability and robustness of models in various learning tasks.

    .. grid-item-card:: Attackers
        :link: binary_classification/attackers.html
        :shadow: md

        Techniques and strategies to simulate adversarial attacks and test model robustness.

.. toctree::
   :hidden:

    binary_classification/metrics
    binary_classification/attackers
