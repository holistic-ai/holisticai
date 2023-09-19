import numpy as np


def unconventional_numeric_config():
    p_attr = np.array(["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"])
    y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]) + 1
    y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1]) + 1
    return p_attr, y_pred, y_true


def numeric_config():
    p_attr = np.array(["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"])
    y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])  # , dtype=np.float32)
    y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1])  # , dtype=np.float32)
    return p_attr, y_pred, y_true


def numeric_nan_config(y_pred_nan=True):
    p_attr = np.array(["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"])
    y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1])
    if y_pred_nan:
        y_pred = np.array([0, 1, 2, 0, 1, 2, np.nan, 1, 2, 0])  # , dtype=np.float32)
    else:
        y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, np.nan, 1])
    return p_attr, y_pred, y_true


def str_config():
    p_attr = np.array(["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"])
    classes = ["a", "b", "c"]
    i2c = dict(zip(range(len(classes)), classes))
    y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([i2c[y] for y in y_pred])
    y_true = np.array([0, 1, 1, 0, 1, 0, 2, 1, 2, 1])
    y_true = np.array([i2c[y] for y in y_true])
    return p_attr, y_pred, y_true
