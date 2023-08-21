from holisticai.utils.models.parameters import ASSESSMENTS
from holisticai.utils.models.feature_importance.wrappers.tensorflow_base.generic import TFBaseModel

class TFBinaryClassificationModel(TFBaseModel):
    def __init__(self, *args,  **kargs):
        super().__init__(*args,  **kargs)
        self._estimator_type='classifier'

    def predict(self, X):
        return self.model.predict(X, verbose=0).round()

    def _predict_proba(self, X): 
        return self.model.predict(X, verbose=0)

    def mode(self, risk_mode):
        if risk_mode==ASSESSMENTS.EFFICACY:
            self.predict_proba = self._predict_proba
