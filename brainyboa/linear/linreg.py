from ..metrics import *
import numpy as np
from .exc import RegressionError

class LinearRegressor:
    def __init__(self, loss_function = 'mean_square_error'):
        self.loss_function = eval(loss_function)
        self.fitted = False

    def fit(self, data_x, data_y):
        x = np.matrix(data_x)
        y = np.matrix(data_y)
        if y.shape[0] == 1:
            y = y.T
        data_size, feature_size = x.shape
        xm = np.c_[np.ones((data_size, 1)), x]
        self.ms = np.linalg.inv(xm.T.dot(xm)).dot(xm.T).dot(y)
        self.fitted = True
        self.coeffs = np.asarray(self.ms[1:]).reshape(-1)
        self.intercept = np.asarray(self.ms[0]).reshape(-1)

    def regress(self, test_x):
        if not self.fitted:
            raise RegressionError("Fit the data before any predictions")
        mat_x = np.matrix(test_x)
        data_size = mat_x.shape[0]
        mat_x = np.c_[np.ones((data_size, 1)), mat_x]
        pred = mat_x.dot(self.ms)
        return pred
