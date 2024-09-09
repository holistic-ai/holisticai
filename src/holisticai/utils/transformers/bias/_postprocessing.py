import inspect

import numpy as np
import pandas as pd

from holisticai.utils._validation import _check_valid_y_proba
from holisticai.utils.obj_rep.object_repr import BMReprObj
from holisticai.utils.transformers._transformer_base import BMTransformerBase


class BMPostprocessing(BMTransformerBase, BMReprObj):
    """
    Base Post Processing transformer
    """

    BM_NAME = "Postprocessing"

    def _load_data(self, **kargs):
        """Save postprocessing atributes and convert data to standard format parameters."""

        params = {}

        if self._has_valid_argument(kargs, "y"):
            y = np.array(kargs.get("y")).ravel()
            params.update({"y": y})

        if self._has_valid_argument(kargs, "y_proba"):
            y_proba = np.array(kargs.get("y_proba"))

            _check_valid_y_proba(y_proba=y_proba)

            params.update({"y_proba": y_proba})

            nb_classes = y_proba.shape[1]
            binary_classes = 2
            if nb_classes == binary_classes:
                favorable_index = 1
                y_score = np.array(y_proba[:, favorable_index]).ravel()
                params.update({"y_score": y_score})

            y_pred = np.argmax(y_proba, axis=1).ravel()
            params.update({"y_pred": y_pred})

        if self._has_valid_argument(kargs, "y_pred"):
            y = np.array(kargs.get("y_pred")).ravel()
            params.update({"y_pred": y})

        params_to_numpy_format = ["group_a", "group_b", "y_score"]
        for param_name in params_to_numpy_format:
            if self._has_valid_argument(kargs, param_name):
                for group_param in ["group_a", "group_b"]:
                    if any(isinstance(kargs.get(group_param), dtype) for dtype in [pd.Series, np.ndarray]):
                        message = f"{group_param} must be a numpy array or pandas series"
                        raise TypeError(message)
                    if kargs.get(group_param).dtype != bool:
                        message = f"{group_param} must be a boolean array"
                        raise TypeError(message)
                    kargs.update({group_param: np.array(kargs.get(group_param))})
                params.update({param_name: self._to_numpy(kargs, param_name)})

        if self._has_valid_argument(kargs, "X"):
            params.update({"X": self._to_numpy(kargs, "X", ravel=False)})

        if self._has_valid_argument(kargs, "sample_weight"):
            params.update({"sample_weight": self._to_numpy(kargs, "sample_weight")})

        elif "y" in kargs:
            params.update({"sample_weight": np.ones_like(params["y"]).astype(np.float64)})

        return params

    @staticmethod
    def _has_valid_argument(kargs, name):
        return (name in kargs) and (kargs[name] is not None)

    def repr_info(self):
        inputs = []
        for p in inspect.signature(self.__init__).parameters:
            try:
                inputs.append(f"{p}={getattr(self,p)}")
            except:  # noqa: E722, S112
                continue
            if len(inputs) == 4:
                inputs.append("...")
                break

        return {
            "dtype": self.__class__.__name__,
            "subtitle": self.__class__.__name__ + "(" + ", ".join(inputs) + ")",
            "attributes": {
                "Type": "Bias Mitigation Postprocessing",
            },
        }
