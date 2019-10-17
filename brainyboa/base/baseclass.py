from brainyboa.metrics import *

class BaseModel:
    def __init__(self):
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True

class BaseClassifier(BaseModel):
    def __init__(self):
        super().__init__()

    def classify(self, X):
        if not self.fitted:
            raise Exception('Model not fitted')

    def score(self, X, y, metric = 'acc_score'):
        pred = self.classify(X)
        return eval(metric)(pred, y)

class BaseRegressor(BaseModel):
    def __init__(self):
        self.fitted = False

    def regress(self, X):
        if not self.fitted:
            raise Exception('Model not fitted')

    def score(self, X, y, metric = 'rsq_score'):
        pred = self.regress(X)
        return eval(metric)(pred, y)
