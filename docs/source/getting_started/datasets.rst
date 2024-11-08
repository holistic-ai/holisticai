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

For example, if we want to load the Adult dataset, we can use the following code:

.. code-block:: python

    from holisticai.datasets import load_hai_datasets
    
    data, target = load_hai_datasets(dataset_name="adult")

.. _processed_datasets:

Processed Datasets
~~~~~~~~~~~~~~~~~~

Processed datasets are refined and structured for specific machine learning tasks, addressing various technical concerns such as bias, efficacy, explainability, etc. These datasets are encapsulated within a :ref:`datasets_objects`, containing variables that are ready for the machine learning process. The table below provides details on these processed datasets. The function load_dataset in :ref:`dataset_loading_functions` allow us to load the processed datasets. The function receibe the following parameters:

- **Protected Attribute**: The attribute that is considered sensitive and should be protected from bias.
- **Processed**: The method used to process X and y. Normally categorical to numerical encoding, normalization, and standardization.
- **Target**: If the dataset has more than one target, the primary target is specified here.

.. csv-table:: Processed Datasets
    :header: "dataset_name", "Dataset", "Learning Task", "protected_attribute"
    :file: datasets.csv
    :widths: 7, 7, 7, 7


You can use this processed datasets using the function load_dataset from holisticai.datasets. For example, to load the processed version of the Adult dataset and use the protected attribute sex, represented by group_a and group_b, we can use the following code:

.. code-block:: python

   from holisticai.datasets import load_dataset
    
    dataset = load_dataset(dataset_name="adult", preprocessed=True, protected_attribute="sex")