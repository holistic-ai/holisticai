# BMBench: an Empirical Bias Mitigation Benchmark for Multitask Machine Learning Predictions

## Overview

BMBench is an empirical benchmark framework designed to evaluate the effectiveness of bias mitigation methods in machine learning systems. It provides a systematic and standardized way to measure and compare the performance of different bias mitigation techniques.

## Evaluation Metrics

| Equality of Opportunity | Equality of Outcome |
|-------------------------|---------------------|

## (Baseline) Bias Mitigation Methods

| Preprocessing | Inprocessing | Postprocessing |
|---------------|--------------|-----------------|

## Features

- **Comprehensive Benchmarking:** Evaluate bias mitigation methods across various machine learning tasks and datasets.
  
- **Standardized Metrics:** Utilize a set of standardized metrics to measure the impact of bias mitigation on different aspects of model performance.
  
- **Easy Integration:** Seamlessly integrate HAIBENCH into your machine learning pipeline to assess and improve the fairness of your models.

## Getting Started

To get started with HAIBENCH, follow these steps:

1. **Simple Usage:**

    ```python
    # install holisticai library
    pip install holisticai

    # import the available tasks and the task framework
    from holisticai.benchmark.tasks import task_name, get_task

    # check the available tasks
    print(task_name)

    # instantiate a task
    task = get_task(task_name='binary_classification')

    # check the actual benchmark results by type of bias mitigation
    task.benchmark(type='preprocessing')

    # instantiate your bias mitigator
    my_mitigator = MyMitigator()

    # run benchmark based on your mitigator and the type of bias mitigation
    task.run_benchmark(mitigator = my_mitigator, type = 'preprocessing')

    # compare the performance of your mitigator with baseline methods
    task.evaluate_table()

    # plot your results (you can select one specific metric to plot)
    task.evaluate_plot(metric = 'Statistical Parity')
    ```
   
## Next Steps

- **Leaderboard:** we are working on a leaderboard to showcase the performance of different bias mitigation methods across various tasks and datasets.

- **Contributions:** we welcome contributions from the community to expand the benchmark framework and improve the evaluation metrics.
