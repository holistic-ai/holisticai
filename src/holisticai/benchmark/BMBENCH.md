# Empirical Benchmarking of Algorithmic Fairness in Machine Learning Models

## Motivation

The development and assessment of bias mitigation methods require rigorous benchmarks. This paper introduces BMBench, a comprehensive benchmarking framework to evaluate bias mitigation strategies across multitask machine learning predictions (binary classification, multiclass classification, regression, and clustering). Our benchmark leverages state-of-the-art and proposed datasets to improve fairness research, offering a broad spectrum of fairness metrics for a robust evaluation of bias mitigation methods. We provide an open-source repository to allow researchers to test and refine their bias mitigation approaches easily, promoting advancements in creating fair machine learning models.

## Get started

### Installation and usage

```bash
pip install holisticai
```

```python
from holisticai.benchmark import BiasMitigationBenchmark

stage = # preprocessing, inprocessing, postprocessing
task = # binary_classification, multiclass, regression, clustering

benchmark = BiasMitigationBenchmark(task, stage)

my_mitigator = MyMitigator()

my_results = benchmark.run(custom_mitigator=my_mitigator)
benchmark.submit()
```

There are many other functionalities available in the `BiasMitigationBenchmark` class. Please refer to the [API documentation](https://holisticai.github.io/holistic-ai/).


## How to cite

If you use BMBench in your research, please cite the following paper:

```bibtex
@article{dacosta2025bmbench,
  title={BMBENCH: Empirical Benchmarking of Algorithmic Fairness in Machine Learning Models},
  author={da Costa, Kleyton and Munoz, Cristian and Fernandez, Franklin and Modenesi, Bernardo and Kazim, Emre and Koshiyama, Adriano},
  url={},
  year={2025}
}
```
