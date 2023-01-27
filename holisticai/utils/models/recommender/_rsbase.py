from typing import Optional

import numpy as np
import pandas as pd

from .item_selection.selectors import ConventionalItemsSelection


class RecommenderSystemBase:
    """
    Recommender System Base Transformer
    """

    def predict(self, X: Optional[np.ndarray], top_n: Optional[int] = 10):

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
