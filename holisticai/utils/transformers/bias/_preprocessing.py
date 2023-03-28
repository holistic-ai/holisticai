from typing import Optional, Union

import numpy as np
import pandas as pd

from holisticai.utils.transformers._transformer_base import BMTransformerBase


class BMPreprocessing(BMTransformerBase):
    """
    Base Preprocessing transformer
    """

    BM_NAME = "Preprocessing"

    def _load_data(self, **kargs):
        """Save postprocessing atributes and convert data to standard format parameters."""

        params = {}
        if "y_true" in kargs:
            y_true = np.array(kargs.get("y_true")).ravel()
            params.update({"y_true": y_true})

        params_to_numpy_format = ["group_a", "group_b"]
        for param_name in params_to_numpy_format:
            if param_name in kargs:
                params.update({param_name: self._to_numpy(kargs, param_name)})

        if "X" in kargs:
            params.update({"X": self._to_numpy(kargs, "X", ravel=False)})

        if ("sample_weight" in kargs) and (kargs["sample_weight"] is not None):
            params.update({"sample_weight": self._to_numpy(kargs, "sample_weight")})

        elif "y_true" in locals():
            params.update({"sample_weight": np.ones_like(y_true).astype(np.float64)})

        elif "X" in kargs:
            params.update(
                {"sample_weight": np.ones(len(params["X"])).astype(np.float64)}
            )

        return params
