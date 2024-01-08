## 0.7.3 (2024-01-08)

### Fix
- Add random state in calibrated equalized odds
- Update Explainability Metrics and Plots
- Refactoring Bias Datasets, Metrics and Mitigation Tutorials

## 0.7.2 (2023-10-25)

### Fix
- Fix explainers metric computation
- Update Tutorials and readthedocs

## 0.7.1 (2023-10-13)

### Fix
- Update readthedocs code
- Update pre-commit config

## 0.7.0 (2023-10-03)

### Feat
- New Technical Risk Module: Explainability
    - Global Metrics based on: Permutation Feature Importance and Surrogate Feature Importance
    - Local Metrics based on: Lime and Shap
- Individual Bias Metrics for classification and regression.

### Fix
- Improving PluginEstimationCalibration and update docstrings
- Clean code in Bias Mitigators

## 0.6.0 (2023-07-14)

### Feat
- Update class and methods documentation.
- Speedup wasserstein barycenters method.
- Add new mode installation: default(only metrics) and methods(plots and mitigators)

### Fix
- Resolve conflicts with sklearn>=1.2.2

## 0.5.0 (2023-06-27)

### Feat
- Update readthedocs project.
- Update plot functions to accepts axis a input arguments.
- Implement Correlation matrix for visualization.
- Resolve conflicts for pandas>=2.0.0 and numpy>=1.24.3

### Fix
- Resolve conflicts with python 3.11
- Update notebook bia tutorials and dataset repository.
- Update documentation for PluginEstimationAndCalibration.

## 0.4.0 (2023-05-18)

### Feat
- Added new detailed tutorials for regression, clustering, multiclassification and recommender system.
- Added efficacy metrics for (#33):
    - binary-classification
    - multiclassification
    - regression
    - clustering
- Added Minimal Cluster Modification for Fairnes (MCMF) method (#26)
- Added prediction function and update tutorial for Fair Top-K Clustering (#23)

### Fix

- added compatibility with sklearn 1.2.1 (#31)

## 0.3.0 (2023-01-27)

### Feat

- release new mitigation techniques (#4)

## 0.2.0 (2022-11-09)

### Feat

- Added binary classification bias metrics:
    - z_test_diff: z-score statistic for statistical parity (2SD rule)
    - z_test_ratio: z-score statistic for disparate impact (2SD rule for ratio)
- Added pre-processing bias mitigation techniques:
    - Correlation Remover (Regression, Multiclass and Binary classification)
    - Fairlet Clustering (Regression)
        - Vanilla
        - Scalable
- Added in-processing bias mitigation techniques:
    - Prejudice Remover (Regression)
    - Meta adversarial debiasing [pytorch] (Binary classification)
    - Grid Search
        - Bounded group loss (Regression)
    - Variational Fair Clustering (Clustering)
    - Fairlet Clustering (Clustering)
        - Vanilla
        - Scalable
    - k-center clustering (Clustering)
    - k-median clustering (Clustering)
        - Local Search
        - Genetic Algorithm
    - Meta fair classifier (Binary and Regression)
- Added post-processing bias mitigation techniques:
    - Wasserstein Barycenters (Regression)
    - Plugin Estimator and Recalibration (Regression)
    - ML debiaser (Binary and Multiclass classification)
    - PL Debiaser (Binary and Multiclass classification)
- Added new dataset: US crime
- Updated the recommender systems tutorial for measuring bias

## 0.1.2 (2022-09-26)

### Docs

- Updated ReadMe file
- Added docs folder
- Added support of Read the docs

## 0.1.1 (2022-09-21)

### Patch

- add bias metrics (holisticai.bias.metrics)
- add mitigation techniques (holisticai.bias.mitigation)
- add bias plotting tools (holisticai.bias.plots)
- add tutorials for measuring bias (tutorials/measuring_bias_tutorials)
- add tutorials for mitigating bias (tutorials/mitigating_bias_tutorials)

## 0.1.0 (2022-09-20)

### Feat

- create repository
