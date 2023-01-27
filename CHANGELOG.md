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
