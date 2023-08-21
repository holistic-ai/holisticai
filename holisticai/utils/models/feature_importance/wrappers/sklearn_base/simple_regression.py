from holisticai.utils.models.feature_importance.wrappers.sklearn_base.generic import WBaseModel

class WSimpleRegressionModel(WBaseModel):
    def __init__(self, *args,  **kargs):
        super().__init__(*args,  **kargs)
        self._estimator_type='regressor'