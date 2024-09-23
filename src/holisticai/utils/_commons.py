from __future__ import annotations

import pandas as pd


def concatenate_metrics(results: dict[str, pd.DataFrame]):
    series = [result.loc[:, "Value"].rename(model_name) for model_name, result in results.items()]
    reference_series = next(iter(results.values())).loc[:, "Reference"]
    series.append(reference_series.rename("Reference"))
    return pd.concat(series, axis=1)
