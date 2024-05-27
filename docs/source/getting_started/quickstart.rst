============
Quickstart
============

Here is a quick start guide to get you up and running with the `holisticai` package.

Bias metrics
-----

.. code-block:: python

  # imports
  from holisticai.metrics.bias import classification_bias_metrics
  from holisticai.datasets import load_dataset
  from holisticai.plots.bias import bias_metrics_report
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

Bias mitigation 
-----

.. code-block:: python

  # imports
  from holisticai.mitigation.inprocessing import Reweighing
  from holisticai.datasets import make_classification()

  # load a toy dataset
  data = make_classification()
