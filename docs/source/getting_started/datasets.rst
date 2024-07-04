========
Datasets
========

In this section, we provide an overview of the datasets used in our projects. The datasets are categorized into two groups: Raw Datasets and Processed Datasets.

.. contents:: Table of Contents
   :local:
   :depth: 1

Raw Datasets
~~~~~~~~~~~~

Raw datasets are the initial data collected from the original sources. These datasets are typically unprocessed and require significant preparation before they can be used for machine learning tasks. The table below provides a summary of the raw datasets.

.. csv-table:: Raw Datasets
    :header: "Dataset", "Method", "Description"
    :file: raw_datasets.csv
    :widths: 7, 7, 15

.. _processed_datasets:

Processed Datasets
~~~~~~~~~~~~~~~~~~

Processed datasets are refined and structured for specific machine learning tasks, addressing various technical concerns such as bias, efficacy, and explainability. These datasets are encapsulated within a Dataset object, containing variables that are ready for the machine learning process. The table below provides details on these processed datasets.

.. csv-table:: Processed Datasets
    :header: "Dataset", "Load Data Method", "Learning Task", "Technical Risk"
    :file: datasets.csv
    :widths: 7, 7, 7, 7