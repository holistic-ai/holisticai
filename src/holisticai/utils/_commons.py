from __future__ import annotations

import numpy as np
import pandas as pd


def concatenate_metrics(results: dict[str, pd.DataFrame]):
    series = [result.loc[:, "Value"].rename(model_name) for model_name, result in results.items()]
    reference_series = next(iter(results.values())).loc[:, "Reference"]
    series.append(reference_series.rename("Reference"))
    return pd.concat(series, axis=1)


def get_number_of_feature_above_threshold_importance(feature_importances, alpha=0.8) -> int:
    feature_importance = np.array(feature_importances)
    ranking_feature_index = np.argsort(feature_importance)[::-1]
    feature_weight = feature_importance[ranking_feature_index] / feature_importance.sum()
    accum_feature_weight = feature_weight.cumsum()
    threshold = max(accum_feature_weight.min(), alpha)
    threshold_index = int(np.argmax(accum_feature_weight >= threshold))
    if accum_feature_weight[threshold_index] < threshold and threshold_index + 1 < len(feature_weight):
        threshold_index += 1
    return threshold_index + 1


def get_item(obj, key):
    if isinstance(obj, pd.DataFrame):
        return obj.loc[key]
    if isinstance(obj, pd.Series):
        return obj.loc[key]
    if isinstance(obj, np.ndarray):
        return obj[key]
    raise ValueError(f"Type {type(obj)} not supported")


def get_columns(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.columns
    if isinstance(obj, np.ndarray):
        return [f"feature_{i}" for i in range(obj.shape[1])]
    raise ValueError(f"Type {type(obj)} not supported")
