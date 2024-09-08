import inspect

import numpy as np

from holisticai.utils.obj_rep.object_repr import BMReprObj
from holisticai.utils.transformers._transformer_base import BMTransformerBase


class BMPreprocessing(BMTransformerBase, BMReprObj):
    """
    Base Preprocessing transformer
    """

    BM_NAME = "Preprocessing"

    def _load_data(self, **kargs):
        """Save postprocessing atributes and convert data to standard format parameters."""

        params = {}
        if "y" in kargs:
            y = np.array(kargs.get("y")).ravel()
            params.update({"y": y})

        params_to_numpy_format = ["group_a", "group_b"]
        for param_name in params_to_numpy_format:
            if param_name in kargs:
                params.update({param_name: self._to_numpy(kargs, param_name).astype(dtype=bool)})

        if "X" in kargs:
            params.update({"X": self._to_numpy(kargs, "X", ravel=False)})

        if ("sample_weight" in kargs) and (kargs["sample_weight"] is not None):
            params.update({"sample_weight": self._to_numpy(kargs, "sample_weight")})

        elif "y" in kargs:
            params.update({"sample_weight": np.ones_like(params["y"]).astype(np.float64)})

        elif "X" in kargs:
            params.update({"sample_weight": np.ones(len(params["X"])).astype(np.float64)})

        return params

    def repr_info(self):
        inputs = []
        for p in inspect.signature(self.__init__).parameters:
            try:
                inputs.append(f"{p}={getattr(self,p)}")
            except:  # noqa: E722, S112
                continue
            if len(inputs)==4:
                inputs.append("...")
                break

        return {
            "dtype": self.__class__.__name__,
            "subtitle": self.__class__.__name__ + "(" + ", ".join(inputs) + ")",
            "attributes": {
                "Type": "Bias Mitigation Preprocessing",
            },
        }