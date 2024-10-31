from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from holisticai.typing import ArrayLike, MatrixLike
from holisticai.utils import Importances, PartialDependence
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def get_fluctuations_from_individuals(grid_values: ArrayLike, individuals: MatrixLike):
    X = grid_values
    fluctuation_index_normalized = []
    for Y in individuals:
        interpolacion = interp1d(X, Y, kind="linear")
        X_new = np.linspace(min(X), max(X), 100)
        Y_new = interpolacion(X_new)
        derivate = np.diff(Y_new)
        sign_changed = np.sum(np.diff(np.sign(derivate)) != 0)
        fluctuation_index_normalized.append(sign_changed / len(Y_new))
    return fluctuation_index_normalized


def fluctuation_ratio(
    partial_dependencies: PartialDependence,
    importances: Optional[Importances] = None,
    top_n=-1,
    label=0,
    weighted=False,
    aggregated=True,
):
    """

    Calculate the fluctuation ratio for features based on partial dependencies.

    Parameters
    ----------
    partial_dependencies : PartialDependence
        An object containing partial dependence values for features.
    importances : Optional[Importances], default=None
        A dictionary or similar structure containing feature importances. Required if `weighted` is True.
    top_n : int, default=-1
        The number of top features to consider. If -1, all features are considered.
    label : int, default=0
        The label for which the partial dependencies are calculated.
    weighted : bool, default=False
        If True, the fluctuation ratios are weighted by feature importances.
    aggregated : bool, default=True
        If True, return the aggregated fluctuation ratio. If False, return a DataFrame with individual fluctuation ratios.

    Returns
    -------
    float or pd.DataFrame
        The aggregated fluctuation ratio if `aggregated` is True. Otherwise, a DataFrame with individual fluctuation ratios.
    """
    feature_names = partial_dependencies.feature_names
    fluctuations = []
    selected_feature_names = feature_names[:top_n]
    for feature_name in selected_feature_names:
        individuals = partial_dependencies.get_value(feature_name=feature_name, label=label, data_type="individual")
        grid_values = partial_dependencies.get_value(feature_name=feature_name, label=label, data_type="grid_values")
        fluctuation_index_normalized = get_fluctuations_from_individuals(grid_values, individuals)
        score = np.mean(fluctuation_index_normalized)
        fluctuations.append(score)
    if aggregated:
        if weighted:
            if importances is None:
                logger.warning(
                    "Importances are required for weighted fluctuation ratio. Using unweighted fluctuation ratio."
                )
                return np.mean(fluctuations)
            return np.sum([importances[fn] * fr for fr, fn in zip(fluctuations, feature_names)])
        return np.mean(fluctuations)
    return pd.DataFrame(fluctuations, columns=["Fluctuation Ratio"], index=selected_feature_names)
