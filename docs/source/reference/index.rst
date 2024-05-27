.. _api:

=============
API reference
=============

This page gives an overview of all public ``holisticai`` objects, functions and
methods. All classes and functions exposed in ``holisticai.*`` namespace are public.

Following is the list of modules in ``holisticai``:

.. grid:: 3 3 3 3
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Metrics
        :link: metrics/index.html
        :shadow: md

        Implement metrics for bias, explainability, efficacy, security, and robustness.

    .. grid-item-card:: Mitigation
        :link: mitigation/index.html
        :shadow: md

        Implement bias mitigation strategies.
    
    .. grid-item-card:: Plots
        :link: plots/index.html
        :shadow: md

        Implement plots and visualizations for bias, explainability, efficacy, security, and robustness.

    .. grid-item-card:: Pipeline
        :link: pipeline.html
        :shadow: md

        Easy framework to build and deploy machine learning pipelines.

    .. grid-item-card:: Datasets
        :link: datasets.html
        :shadow: md

        Datasets for experiments and testing.

.. If you update this toctree, also update the manual toctree in the
.. main index.rst.template

.. toctree::
   :hidden:

   metrics/index
   mitigation/index
   plots/index
   pipeline
   datasets