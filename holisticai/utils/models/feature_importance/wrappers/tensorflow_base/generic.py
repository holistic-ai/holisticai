class TFBaseModel:
    def __init__(self, model):
        self.model = model

    def mode(self, risk_mode):
        pass

    def predict(self, *args, **kargs): return self.model.predict(*args, **kargs)