class BaseFunction:
    def __init__(self, func = None):
        self.func = func
        self.deriv = None

    def set_deriv(self, func):
        self.deriv = func

    def evaluate(self, params):
        return self.func(params)

    def evaluate_deriv(self, params):
        return self.deriv(params)
