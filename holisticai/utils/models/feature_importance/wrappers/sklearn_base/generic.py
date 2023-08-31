class WBaseModel:
    def __init__(self, model):
        self.model = model

    def __sklearn_is_fitted__(self):
        return True

    def mode(self, risk_mode):
        pass

    def predict(self, *args, **kargs):
        return self.model.predict(*args, **kargs)

    def fit(self, *args, **kargs):
        return self.model.fit(*args, **kargs)

    def score(self, *args, **kargs):
        return self.model.score(*args, **kargs)

    def change_mode_single_output(self, out_col):
        raise NotImplementedError()
