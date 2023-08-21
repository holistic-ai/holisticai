from holisticai.utils.models.feature_importance.wrappers.sklearn_base.generic import WBaseModel

class WBinaryClassificationModel(WBaseModel):
    def __init__(self, *args,  **kargs):
        super().__init__(*args,  **kargs)
        self._estimator_type='classifier'
        self.classes_ = self.model.classes_

    def mode(self, risk_mode):
        if risk_mode=='explainability':
            self.predict_proba = self.model.predict_proba

        elif risk_mode=='efficacy':
            self.predict_proba = self._predict_proba

    def _predict_proba(self, X):
        baseline_ypred_proba = self.model.predict_proba(X)
        baseline_ypred_proba = baseline_ypred_proba[:, 1]
        return baseline_ypred_proba
