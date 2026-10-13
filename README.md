<p align="center">
  <img src="docs/source/holistic_ai.png" alt="Holistic AI" width="90">
</p>

<h1 align="center">holisticai</h1>

<p align="center"><strong>Measure and mitigate what accuracy does not tell you about a model.</strong></p>

<p align="center">
  <a href="https://pypi.org/project/holisticai/"><img
    alt="Latest version on PyPI"
    src="https://img.shields.io/pypi/v/holisticai?style=flat-square&labelColor=141E41&color=36B1FE"></a>
  <a href="https://pypi.org/project/holisticai/"><img
    alt="Python 3.9 or newer"
    src="https://img.shields.io/badge/python-3.9+-36B1FE?style=flat-square&labelColor=141E41"></a>
  <a href="https://pypi.org/project/holisticai/"><img
    alt="Downloads per month"
    src="https://img.shields.io/pypi/dm/holisticai?style=flat-square&labelColor=141E41&color=36B1FE"></a>
  <a href="https://holisticai.readthedocs.io/en/latest/"><img
    alt="Documentation"
    src="https://img.shields.io/badge/docs-readthedocs-36B1FE?style=flat-square&labelColor=141E41"></a>
  <a href="LICENSE"><img
    alt="Apache 2.0 licence"
    src="https://img.shields.io/badge/license-Apache--2.0-36B1FE?style=flat-square&labelColor=141E41"></a>
  <a href="https://join.slack.com/t/holisticaicommunity/shared_invite/zt-2jamouyrn-BrMfeoBZIHT8HbLzB3P9QQ"><img
    alt="Join the community on Slack"
    src="https://img.shields.io/badge/slack-join-AB5FCE?style=flat-square&labelColor=141E41"></a>
</p>

<p align="center">
  <a href="https://holisticai.readthedocs.io/en/latest/">Documentation</a> ·
  <a href="https://holisticai.readthedocs.io/en/latest/getting_started/index.html">Getting started</a> ·
  <a href="https://holisticai.readthedocs.io/en/latest/reference/index.html">API reference</a> ·
  <a href="https://holisticai.readthedocs.io/en/latest/gallery/index.html">Gallery</a> ·
  <a href="tutorials">Notebooks</a> ·
  <a href="https://holisticai.com">holisticai.com</a>
</p>

An open-source library for assessing and improving the safety aspects of AI
systems. It answers five questions a validation accuracy cannot: does the model
treat groups differently, can its decisions be accounted for, does it hold up
under shift and attack, what does it leak about its training data, and is it
actually effective. Each answer is a metric you can compute in a few lines
against a scikit-learn-shaped model — and for bias, a mitigator you can drop
into the pipeline that produced it.

## Install

```bash
pip install holisticai
```

The base install carries scikit-learn, pandas, numpy and pyarrow — enough for
every metric that needs nothing more. The heavier machinery is opt-in:

| Extra | Install | Adds |
|---|---|---|
| **datasets** | `pip install holisticai[datasets]` | The dataset loaders, plus matplotlib and seaborn for every plot |
| **bias** | `pip install holisticai[bias]` | The 28 bias mitigators — JAX, optax, flax and pygad |
| **explainability** | `pip install holisticai[explainability]` | LIME and SHAP feature-importance backends |
| **security** | `pip install holisticai[security]` | JAX and jaxopt for the privacy attacks |
| **all** | `pip install holisticai[all]` | Everything above |

Needs Python 3.9 or newer. On macOS, the solvers behind some bias mitigators
want a C toolchain first:

```bash
brew install cbc pkg-config cmake
python -m pip install cylp
```

## Measure

Train anything with a `fit`/`predict` interface, then hand the predictions and
the group membership to a metrics report:

```python
from holisticai.bias.metrics import classification_bias_metrics
from holisticai.bias.plots import bias_metrics_report
from holisticai.datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# load a bundled dataset, split by a protected attribute
dataset = load_dataset("law_school", protected_attribute="race")
split = dataset.train_test_split(test_size=0.3)
train, test = split["train"], split["test"]

# train an ordinary model — the library makes no demands on it
scaler = StandardScaler()
model = LogisticRegression(random_state=42, max_iter=500)
model.fit(scaler.fit_transform(train["X"]), train["y"])
y_pred = model.predict(scaler.transform(test["X"]))

# every relevant bias metric, with a fair reference value for each
metrics = classification_bias_metrics(
    group_a=test["group_a"],
    group_b=test["group_b"],
    y_true=test["y"],
    y_pred=y_pred,
)

bias_metrics_report(model_type="binary_classification", table_metrics=metrics)
```

## Mitigate

`holisticai.pipeline.Pipeline` is scikit-learn's pipeline with a slot for a
mitigator. Group membership reaches it through `bm__` prefixed arguments, so the
rest of your training code is unchanged:

```python
from holisticai.bias.mitigation import Reweighing
from holisticai.pipeline import Pipeline

pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("bm_preprocessing", Reweighing()),
        ("estimator", LogisticRegression()),
    ]
)
pipeline.fit(train["X"], train["y"], bm__group_a=train["group_a"], bm__group_b=train["group_b"])
y_pred = pipeline.predict(test["X"], bm__group_a=test["group_a"], bm__group_b=test["group_b"])
```

Swap `bm_preprocessing` for `bm_inprocessing` or `bm_postprocessing` to mitigate
at a different stage. All 28 mitigators are listed under
[bias mitigation](https://holisticai.readthedocs.io/en/latest/reference/bias/mitigation.html).

## What is inside

| Vertical | What it answers | What it ships |
|---|---|---|
| **Bias** | Does the model treat groups differently? | 65 metrics across binary classification, multiclass, regression, clustering and recommendation · 28 mitigators, pre-, in- and post-processing · 20 plots |
| **Explainability** | Can its decisions be accounted for? | 32 metrics over global and local feature importance, surrogate models and decision trees · LIME and SHAP backends |
| **Robustness** | Does it hold up under shift and attack? | Adversarial accuracy, empirical robustness and accuracy-degradation profiles · 4 attackers — HopSkipJump, ZooAttack and two regression poisoners |
| **Security** | What does it leak about its training data? | 10 metrics — k-anonymity, l-diversity, SHAPr, privacy risk score, attribute-inference and data-minimization · anonymization mitigation |
| **Efficacy** | Is it actually effective? | Batch metric reports for classification, multiclass, regression and clustering |

Every metric is a plain function over arrays, and every batch report returns a
pandas DataFrame — nothing here asks you to restructure your project around it.

### Datasets

`load_dataset` returns a `Dataset` already split into `X`, `y`, `group_a` and
`group_b` for a protected attribute you name, so an example is one line rather
than a preprocessing script. Files are pulled from
[holistic-ai/holisticai-datasets](https://huggingface.co/datasets/holistic-ai/holisticai-datasets)
on first use and cached under `~/holisticai_data` (`HOLISTIC_AI_DATA` moves it):

`adult` · `law_school` · `student` · `student_multiclass` · `us_crime` ·
`us_crime_multiclass` · `german_credit` · `census_kdd` · `bank_marketing` ·
`compas_two_year_recid` · `compas_is_recid` · `diabetes` · `clinical_records` ·
`acsincome` · `acspublic` · `lastfm` · `mw_small` · `mw_medium`

### At a glance

| | |
|---|---|
| **Verticals** | 5 — bias, explainability, robustness, security, efficacy |
| **Metrics** | 65 bias · 32 explainability · 10 security · 7 efficacy · 4 robustness |
| **Mitigation** | 28 bias mitigators · 1 anonymization mitigator · 4 attackers |
| **Datasets** | 18, fetched and cached on first use, split by protected attribute in one call |
| **Base install** | scikit-learn, pandas, numpy, pyarrow — the rest is opt-in |
| **Python** | 3.9+, tested on 3.9 in CI |
| **Licence** | Apache 2.0 |

## Tutorials

[`tutorials/`](tutorials) holds runnable notebooks per vertical — measuring and
mitigating bias by task, measuring robustness and dataset shift, privacy
measurement and mitigation, explainability demos, efficacy, trade-off analysis
and MLflow integration. The same material, rendered, is the
[example gallery](https://holisticai.readthedocs.io/en/latest/gallery/index.html).

## Develop

```bash
git clone https://github.com/holistic-ai/holisticai
cd holisticai
pip install hatch
```

Two commands, both of which CI runs on every pull request. Get them passing
before opening one:

```bash
hatch fmt --check          # ruff lint and format, configured by ruff_default.toml
hatch run testing:run      # pytest over tests/
```

Docs are Sphinx with the pydata theme:

```bash
hatch run sphinx:build     # builds docs/source into docs/build
```

## Contributing

[`CONTRIBUTING.md`](CONTRIBUTING.md) covers setting up an environment and what
makes a change easy to accept. Questions and ideas are welcome in
[issues](https://github.com/holistic-ai/holisticai/issues) or on
[Slack](https://join.slack.com/t/holisticaicommunity/shared_invite/zt-2jamouyrn-BrMfeoBZIHT8HbLzB3P9QQ).

## Contributors

<a href="https://github.com/holistic-ai/holisticai/graphs/contributors">
  <img alt="Contributors to holisticai" src="https://contrib.rocks/image?repo=holistic-ai/holisticai">
</a>

## Licence

Apache License 2.0 — see [LICENSE](LICENSE).
