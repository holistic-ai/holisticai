import numpy as np

from holisticai.utils.transformers._transformer_base import BMTransformerBase


class BMInprocessing(BMTransformerBase):
    """
    Base Inprocessing transformer
    """

    BM_NAME = "Inprocessing"

    def _load_data(self, **kargs):
        """Save postprocessing atributes and convert data to standard format parameters."""

        params = {}
        if "y" in kargs:
            y = np.array(kargs.get("y")).ravel()
            classes_ = list(np.unique(y))
            params.update({"y": y, "classes_": classes_})

        params_to_numpy_format = ["group_a", "group_b"]
        for param_name in params_to_numpy_format:
            if param_name in kargs:
                params.update({param_name: self._to_numpy(kargs, param_name)})

        if "X" in kargs:
            params.update({"X": self._to_numpy(kargs, "X", ravel=False)})

        if ("sample_weight" in kargs) and (kargs["sample_weight"] is not None):
            params.update({"sample_weight": self._to_numpy(kargs, "sample_weight")})

        elif "y" in locals():
            params.update({"sample_weight": np.ones_like(y).astype(np.float64)})

        return params
