import numpy as np

from holisticai.utils._validation import _check_valid_y_proba
from holisticai.utils.transformers._transformer_base import BMTransformerBase


class BMPostprocessing(BMTransformerBase):
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
                params.update({param_name: self._to_numpy(kargs, param_name)})

        if self._has_valid_argument(kargs, "X"):
            params.update({"X": self._to_numpy(kargs, "X", ravel=False)})

        if self._has_valid_argument(kargs, "sample_weight"):
            params.update({"sample_weight": self._to_numpy(kargs, "sample_weight")})

        elif "y" in locals():
            params.update({"sample_weight": np.ones_like(y).astype(np.float64)})

        return params

    @staticmethod
    def _has_valid_argument(kargs, name):
        return (name in kargs) and (kargs[name] is not None)
