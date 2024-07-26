from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from holisticai.security.commons.data_minimization._modificators import ModifierHandler, ModifierType
from holisticai.security.commons.data_minimization._selectors import SelectorsHandler, SelectorType
from holisticai.utils import ModelProxy

logger = logging.getLogger(__name__)


class DataMinimizer:
    def __init__(
        self,
        proxy: ModelProxy,
        selector_types: Optional[list[SelectorType]] = None,
        modifier_types: Optional[list[ModifierType]] = None,
    ):
        if modifier_types is None:
            modifier_types = ["Average", "Permutation"]
        if selector_types is None:
            selector_types = ["FImportance", "Percentile", "Variance"]
        self.proxy = proxy
        self.selector_types = selector_types
        self.modifier_types = modifier_types
        self.shdl = SelectorsHandler(proxy=self.proxy, selector_types=selector_types)
        self.modifier = ModifierHandler(methods=modifier_types)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.shdl.fit(X, y)

    def predict(self, X):
        y_pred_dm = []
        for sn, selector in self.shdl:
            logger.info(f"Transforming X using selector {sn}")
            indexes = selector.get_feature_names_out()
            xts = self.modifier(X, indexes)
            for mn, mod in xts.items():
                pred = self.proxy.predict(mod["x"])
                n_feats = len(mod["updated_features"])
                if n_feats > 0:
                    y_pred_dm.append(
                        {
                            "selector_type": sn,
                            "modifier_type": mn,
                            "n_feats": n_feats,
                            "feats": mod["updated_features"],
                            "predictions": pred,
                        }
                    )
        return y_pred_dm
