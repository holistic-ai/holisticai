<h1 align="center">
<img src="docs/source/holistic_ai.png" width="100">
<br>holisticai: building trustworthy AI systems
</h1>

[![PyPI](https://img.shields.io/pypi/v/holisticai)](https://pypi.org/project/holisticai/)
[![Documentation Status](https://readthedocs.org/projects/holisticai/badge/?version=latest)](https://holisticai.readthedocs.io/en/latest/?badge=latest)
[![PyPI - License](https://img.shields.io/github/license/holistic-ai/holisticai)](https://img.shields.io/github/license/holistic-ai/holisticai)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/holisticai)](https://img.shields.io/pypi/dm/holisticai)
[![Slack](https://img.shields.io/badge/Slack-Join-blue)](https://join.slack.com/t/holisticaicommunity/shared_invite/zt-2jamouyrn-BrMfeoBZIHT8HbLzB3P9QQ)

---

Holistic AI is an open-source library dedicated to assessing and improving the trustworthiness of AI systems. We believe that responsible AI development requires a comprehensive evaluation across multiple dimensions, beyond just accuracy.

### Current Capabilities
---

Holistic AI currently focuses on five verticals of AI trustworthiness: 

1. **Bias:** measure and mitigate bias in AI models.
2. **Explainability:** measure into model behavior and decision-making.
3. **Robustness:** measure model performance under various conditions.
4. **Security:** measure the privacy risks associated with AI models.
5. **Efficacy:** measure the effectiveness of AI models.


### Quick Start
---

```bash
pip install holisticai  # Basic installation
pip install holisticai[bias]  # Bias mitigation support
pip install holisticai[explainability]  # For explainability metrics and plots
pip install holisticai[all]  # Install all packages for bias and explainability
```

```python
# imports
from holisticai.bias.metrics import classification_bias_metrics
from holisticai.datasets import load_dataset
from holisticai.bias.plots import bias_metrics_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# load an example dataset and split
dataset = load_dataset('law_school')
dataset_split = dataset.train_test_split(test_size=0.3)

# separate the data into train and test sets
train_data = dataset_split['train']
test_data = dataset_split['test']

# rescale the data
scaler = StandardScaler()
X_train_t = scaler.fit_transform(train_data['x'])
X_test_t = scaler.transform(test_data['x'])

# train a logistic regression model
model = LogisticRegression(random_state=42, max_iter=500)
model.fit(X_train_t, train_data['y'])

# make predictions
y_pred = model.predict(X_test_t)

# compute bias metrics
metrics = classification_bias_metrics(
    group_a = test_data['group_a'],
    group_b = test_data['group_b'],
    y_true = test_data['y'],
    y_pred = y_pred
    )

# create a comprehensive report
bias_metrics_report(model_type='binary_classification', table_metrics=metrics)
```

### Key Features
---

* **Comprehensive Metrics:**  Measure various aspects of AI system trustworthiness, including bias, fairness, and explainability.
* **Mitigation Techniques:** Implement strategies to address identified issues and improve the fairness and robustness of AI models.
* **User-Friendly Interface:**  Intuitive API for easy integration into existing workflows.
* **Visualization Tools:**  Generate insightful visualizations for better understanding of model behavior and bias patterns.

### Documentation and Tutorials
---

* [Documentation](https://holistic-ai.readthedocs.io/en/latest/)
* [Notebooks](https://github.com/holistic-ai/holisticai/tree/main/tutorials)
* [Holistic AI Website](https://holisticai.com)


### Detailed Installation
---

**Troubleshooting (macOS):**

Before installing the library, you may need to install these packages:

```bash
brew install cbc pkg-config
python -m pip install cylp
brew install cmake
```

**Explainability Visualization Tools:**

Install GraphViz:

```bash
sudo apt update
sudo apt-get install graphviz
```

### Contributing

We welcome contributions from the community To learn more about contributing to Holistic AI, please refer to our [Contributing Guide](CONTRIBUTING.md).