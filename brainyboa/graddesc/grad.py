import numpy as np
from brainyboa.utils import BaseFunction
from brainyboa.metrics import squared_loss, squared_loss_derivative

class GradientDescentRegressor:
    def __init__(self, loss = 'squared_loss', l_rate = 0.01, iters = 15000, tolerance = 0.0001):
        self.loss = loss
        self.l_rate = l_rate
        self.iters = iters
        self.tolerance = tolerance

    def fit(self, x, y):
        x1 = np.matrix(x)
        sample_size, feature_size = x1.shape
        y1 = np.matrix(y)
        x1 = np.insert(x1, 0, 1, axis = 1)
        self.theta = np.zeros((1, feature_size + 1))

        for i in range(self.iters):
            gradient = eval(self.loss + '_derivative')(x1, y.T, self.theta)
            self.theta -= self.l_rate * gradient
            cost = eval(self.loss)(x1, y, self.theta)
            if cost <= self.tolerance:
                break
        return self.theta

    def regress(self, X):
        X = np.matrix(X)
        X = np.insert(X, 0, 1, axis = 1)
        return np.array(self.theta * X.T).flatten()
