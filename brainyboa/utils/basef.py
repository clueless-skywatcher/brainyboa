class BaseFunction:
    def __init__(self, func = None, params = None):
        self.params = params
        self.func = func

    def derivative(self):
        self.deriv = None
        pass

    def evaluate(self):
        return self.func(self.params)
