import numpy as np
import pandas as pd


class LimeTabularHandler:
    def __init__(self, scorer, *args, **kargs):
        self.scorer = scorer
        from lime import lime_tabular

        self.explainer = lime_tabular.LimeTabularExplainer(*args, **kargs)

    def explain_one_sample(self, x0):
        exp = self.explainer.explain_instance(
            x0, self.scorer, num_features=len(x0), num_samples=200
        )
        exp_values = np.array(list(exp.local_exp.values())[0])
        ranked_imp = exp_values[:, 1]
        rank = exp_values[:, 0].argsort()
        importance = np.zeros_like(ranked_imp)
        importance[rank] = ranked_imp
        return importance

    def __call__(self, Xsel):
        values = np.stack(
            [self.explain_one_sample(x) for index, x in Xsel.iterrows()], axis=0
        )
        importances = abs(values) / abs(values).sum(axis=1, keepdims=True)
        return pd.DataFrame(importances, columns=Xsel.columns, index=Xsel.index)


class ShapTabularHandler:
    def __init__(self, *args, **kargs):
        import shap

        self.explainer = shap.Explainer(*args, **kargs)

    def __call__(self, Xsel):
        values = self.explainer(Xsel).values
        importances = abs(values) / abs(values).sum(axis=1, keepdims=True)
        return pd.DataFrame(importances, columns=Xsel.columns, index=Xsel.index)
