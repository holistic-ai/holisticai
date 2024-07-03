import numpy as np
import pandas as pd
from holisticai.datasets import Dataset


class SHAPImportanceCalculator:
    importance_type: str = "local"

    def __init__(self, x_train, model):
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP is not installed. Please install it using 'pip install shap'") from None

        masker = shap.maskers.Independent(x_train)
        self.explainer = shap.Explainer(model.predict, masker=masker)

    def __call__(self, ds: Dataset):
        shap_values = self.explainer(ds["X"])
        return pd.DataFrame(data=np.abs(shap_values.values), columns=ds["X"].columns)
