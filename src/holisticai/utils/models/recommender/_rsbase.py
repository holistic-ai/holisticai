from __future__ import annotations

import numpy as np
import pandas as pd
from holisticai.utils.models.recommender.item_selection.selectors import ConventionalItemsSelection


class RecommenderSystemBase:
    """
    Recommender System Base Transformer
    """

    def predict(self, X: np.ndarray | None, top_n: int | None = 10):
        """
        Predict recommendations

        Parameters
        ----------
        X : matrix-like
            scored matrix, 0 means non-raked cases.

        top_n : int
            Number of recommendations to return.

        Returns
        -------
        DataFrame
            A DataFrame with recommendations
        """
        self.item_selection = ConventionalItemsSelection(top_n=top_n)
        time_mask = X > 0
        selected_items = self.item_selection(time_mask, self.pred)[0]
        dfs = []
        for query_id, indexes in enumerate(selected_items):
            df = pd.DataFrame()
            df["score"] = np.array(self.pred[query_id, indexes])
            df["Y"] = np.array(indexes)
            df["X"] = query_id
            dfs.append(df)
        dfs = pd.concat(dfs, axis=0)
        return dfs[["X", "Y", "score"]]
