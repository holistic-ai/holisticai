import pandas as pd


class SurrogateBase:
    def __init__(self, **_):
        self._surrogate = None
        self.__is_fitted__ = False

    def build(self, X: pd.DataFrame, y: pd.Series):
        pass

    def fit(self, X, y):
        self._surrogate = self.build(X, y)
        self.__is_fitted__ = True
        return self

    def predict(self, X):
        assert self._surrogate is not None, "Model not fitted"
        return self._surrogate.predict(X)



class SurrogateTreeBase(SurrogateBase):
    pass