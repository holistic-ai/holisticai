from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from holisticai.utils._validation import _array_like_to_numpy

if TYPE_CHECKING:
    from holisticai.typing._typing import ArrayLike


def concatenate_metrics(results: dict[str, pd.DataFrame]):
    series = [result.loc[:, "Value"].rename(model_name) for model_name, result in results.items()]
    reference_series = next(iter(results.values())).loc[:, "Reference"]
    series.append(reference_series.rename("Reference"))
    return pd.concat(series, axis=1)


def get_top_ranking_from_scores(feature_importances: ArrayLike, alpha=0.8):
    feature_importance = _array_like_to_numpy(feature_importances)
    feature_weight = np.sort(feature_importance)[::-1] / feature_importance.sum()
    accum_feature_weight = feature_weight.cumsum()
    threshold = max(accum_feature_weight.min(), alpha)
    threshold_index = np.argmax(accum_feature_weight >= threshold)
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