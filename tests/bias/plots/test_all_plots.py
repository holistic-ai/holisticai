# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Regression Plots
# Recommender Plots
# Multiclass Plots
# Exploratory plots
# Classification Plots
from holisticai.bias.plots import (
    abroca_plot,
    accuracy_bar_plot,
    disparate_impact_curve,
    disparate_impact_plot,
    distribution_plot,
    exposure_diff_plot,
    exposure_ratio_plot,
    frequency_matrix_plot,
    frequency_plot,
    group_pie_plot,
    histogram_plot,
    long_tail_plot,
    mae_bar_plot,
    rmse_bar_plot,
    statistical_parity_curve,
    statistical_parity_plot,
    success_rate_curve,
    success_rate_curves,
)

# Adult df
from holisticai.datasets import load_adult

df = load_adult()["frame"]

# Last fm df
from holisticai.datasets import load_last_fm
from holisticai.utils import recommender_formatter

df_2 = load_last_fm()["frame"]
df_2["score"] = np.ones(len(df_2))
df_pivot, p_attr = recommender_formatter(
    df_2, users_col="user", groups_col="sex", items_col="artist", scores_col="score"
)


def test_abroca_plot(monkeypatch):
    """test_abroca_plot"""
    monkeypatch.setattr(plt, "show", lambda: None)
    y_true = df["class"] == ">50K"
    y_score = np.random.random(len(y_true))
    group_a = df["sex"] == "Male"
    group_b = df["sex"] == "Female"
    abroca_plot(group_a, group_b, y_score, y_true)
    assert True


def test_abroca_plot_aux(monkeypatch):
    """test_abroca_plot auxiliary"""
    monkeypatch.setattr(plt, "show", lambda: None)
    y_true = df["class"] == ">50K"
    y_score = np.random.random(len(y_true))
    group_a = df["sex"] == "Male"
    group_b = df["sex"] == "Female"
    abroca_plot(group_a, group_b, y_score, y_true, ax=None, size=(5, 5), title="TITLE")
    assert True


def test_distribution_plot(monkeypatch):
    """test_distribution_plot"""
    monkeypatch.setattr(plt, "show", lambda: None)
    distribution_plot(
        df["age"], df["education-num"], ax=None, size=(20, 10), title="blabla"
    )
    assert True


def test_group_pie_plot(monkeypatch):
    """test_group_pie_plot"""
    monkeypatch.setattr(plt, "show", lambda: None)
    _, ax = plt.subplots()
    group_pie_plot(np.array(df["occupation"]), ax=ax, title="occupation", size=(10, 10))
    assert True


def test_histogram_plot(monkeypatch):
    """test_histogram_plot"""
    monkeypatch.setattr(plt, "show", lambda: None)
    _, ax = plt.subplots()
    histogram_plot(
        df["education"], p_attr=df["sex"], ax=ax, size=(10, 7), title="BLUBH"
    )
    assert True


def test_frequency_plot(monkeypatch):
    """test_frequency_plot"""
    monkeypatch.setattr(plt, "show", lambda: None)
    frequency_plot(p_attr=df["sex"], y_pred=df["class"])
    assert True


def test_frequency_matrix_plot(monkeypatch):
    """test_frequency_matrix_plot"""
    monkeypatch.setattr(plt, "show", lambda: None)
    _, ax = plt.subplots()
    frequency_matrix_plot(p_attr=df["sex"], y_pred=df["class"], ax=ax)
    assert True


def test_statistical_parity_plot(monkeypatch):
    """test_statistical_parity_plot"""
    monkeypatch.setattr(plt, "show", lambda: None)
    statistical_parity_plot(p_attr=df["sex"], y_pred=df["class"] == ">50K")
    assert True


def test_disparate_impact_plot(monkeypatch):
    """test_disparate_impact_plot"""
    monkeypatch.setattr(plt, "show", lambda: None)
    _, ax = plt.subplots()
    disparate_impact_plot(p_attr=df["sex"], y_pred=df["class"] == ">50K", ax=ax)
    assert True


def test_accuracy_bar_plot(monkeypatch):
    """test_accuracy_bar_plot"""
    monkeypatch.setattr(plt, "show", lambda: None)
    _, ax = plt.subplots()
    accuracy_bar_plot(
        p_attr=df["sex"],
        y_pred=df["class"] == ">50K",
        y_true=df["class"] == ">50K",
        ax=ax,
    )
    assert True


def test_success_rate_curve(monkeypatch):
    """test_success_rate_curve"""
    monkeypatch.setattr(plt, "show", lambda: None)
    group_a = df["sex"] == "Female"
    group_b = df["sex"] == "Male"
    y_pred = df["age"]
    success_rate_curve(np.array(group_a), np.array(group_b), np.array(y_pred))
    assert True


def test_statistical_parity_curve(monkeypatch):
    """test_statistical_parity_curve"""
    monkeypatch.setattr(plt, "show", lambda: None)
    group_a = df["sex"] == "Female"
    group_b = df["sex"] == "Male"
    y_pred = df["age"]
    statistical_parity_curve(np.array(group_a), np.array(group_b), np.array(y_pred))
    assert True


def test_statistical_parity_curve_aux(monkeypatch):
    """test_statistical_parity_curve aux"""
    monkeypatch.setattr(plt, "show", lambda: None)
    group_a = df["sex"] == "Female"
    group_b = df["sex"] == "Male"
    y_pred = df["age"]
    statistical_parity_curve(
        np.array(group_a), np.array(group_b), np.array(y_pred), x_axis="quantile"
    )
    assert True


def test_disparate_impact_curve(monkeypatch):
    """test_disparate_impact_curve"""
    monkeypatch.setattr(plt, "show", lambda: None)
    group_a = df["sex"] == "Female"
    group_b = df["sex"] == "Male"
    y_pred = df["age"]
    disparate_impact_curve(np.array(group_a), np.array(group_b), np.array(y_pred))
    assert True


def test_success_rate_curves(monkeypatch):
    """test_success_rate_curves"""
    monkeypatch.setattr(plt, "show", lambda: None)
    success_rate_curves(df["education"], np.array(df["age"]), size=(10, 10))
    assert True


def test_rmse_bar_plot(monkeypatch):
    """test_rmse_bar_plot"""
    monkeypatch.setattr(plt, "show", lambda: None)
    y_pred = df["class"] == "50K"
    y_true = 1 - y_pred
    rmse_bar_plot(df["education"], y_pred, y_true, ax=None, size=None, title=None)
    assert True


def test_mae_bar_plot(monkeypatch):
    """test_mae_bar_plot"""
    monkeypatch.setattr(plt, "show", lambda: None)
    y_pred = df["class"] == "50K"
    y_true = 1 - y_pred
    mae_bar_plot(df["education"], y_pred, y_true, ax=None, size=None, title=None)
    assert True


def test_long_tail_plot(monkeypatch):
    """test_long_tail_plot"""
    monkeypatch.setattr(plt, "show", lambda: None)
    mat_pred = df_pivot.to_numpy()
    long_tail_plot(mat_pred, top=None, thresh=0.5, normalize=False)
    assert True


def test_long_tail_plot_aux(monkeypatch):
    """test_long_tail_plot auxiliary"""
    monkeypatch.setattr(plt, "show", lambda: None)
    mat_pred = df_pivot.to_numpy()
    long_tail_plot(mat_pred, top=100, thresh=0.5, normalize=False)
    assert True


def test_exposure_diff_plot(monkeypatch):
    """test_exposure_diff_plot"""
    monkeypatch.setattr(plt, "show", lambda: None)
    group_a = p_attr == "f"
    group_b = p_attr == "m"
    mat_pred = df_pivot.to_numpy()
    exposure_diff_plot(
        group_a,
        group_b,
        mat_pred,
        top=None,
        thresh=0.5,
        normalize=False,
        ax=None,
        size=None,
        title=None,
    )
    assert True


def test_exposure_ratio_plot(monkeypatch):
    """test_exposure_ratio_plot"""
    monkeypatch.setattr(plt, "show", lambda: None)
    group_a = p_attr == "f"
    group_b = p_attr == "m"
    mat_pred = df_pivot.to_numpy()
    exposure_ratio_plot(
        group_a,
        group_b,
        mat_pred,
        top=None,
        thresh=0.5,
        normalize=False,
        ax=None,
        size=None,
        title=None,
    )
    assert True
