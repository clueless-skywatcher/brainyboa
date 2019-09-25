import numpy as np
from .basef import BaseFunction

class Sigmoid(BaseFunction):
    def _sigmoid(self, params):
        return 1 / (1 + np.exp(params))
    def _sigmoid_deriv(self, params):
        return self._sigmoid(params) * (1 - self._sigmoid(params))
    def __init__(self, params):
        self.func = self._sigmoid
        self.params = params
    def derivative(self):
        self.deriv = self._sigmoid_deriv
        return self.deriv(self.params)
