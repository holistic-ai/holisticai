Holistic AI Library 
===================

Welcome to the Holistic AI library! This is an open-source tool to assess and improve the trustworthiness of AI systems. The current version of the library offers a set of techniques to easily measure and mitigate Bias across a variety of tasks.  

.. toctree::
    :maxdepth: 1
    :caption: Documentation
    
    metrics
    mitigation
    datasets
    pipeline
    tutorials

Our long-term goal is to offer a set of techniques to easily measure and mitigate AI risks across five  areas: Bias, Efficacy, Robustness, Privacy and Explainability. This will allow a holistic assessment of AI systems. 

Installation
------------

You can install the library with `pip` using the following command:

.. code-block::

  pip install holisticai # basic installation
  pip install holisticai[bias] # bias mitigation support
  pip install holisticai[all] # install all packages for bias and explainability 
