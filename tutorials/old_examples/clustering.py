import os
import sys

import numpy as np

sys.path.append(os.getcwd())
sys.path.append("./../../")
# Get data
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from holisticai.mitigation.bias import FairKmedianClustering
from tutorials.utils.datasets import preprocessed_dataset

train_data, test_data = preprocessed_dataset("adult")
num = 2000
t_data = [np.array(d)[:num] for d in train_data]
X_train, _, group_a_train, group_b_train = t_data
group_a_train = group_a_train.reshape(-1)
group_b_train = group_b_train.reshape(-1)

Xt = StandardScaler().fit_transform(X_train)
model = FairKmedianClustering(
    n_clusters=4, seed=42, verbose=True, strategy="GA", max_iter=500
)
model.fit(Xt, group_a_train, group_b_train)
