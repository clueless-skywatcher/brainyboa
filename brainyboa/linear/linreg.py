from ..metrics import *
import numpy as np
from exc import RegressionError

class LinearRegressor:
    def __init__(self, loss_function = 'mean_square_error'):
        self.loss_function = eval(loss_function)
        self.fitted = False


    def fit(self, data_x, data_y):
        x = np.matrix(data_x)
        y = np.matrix(data_y)
        data_size, feature_size = x.shape
        xm = np.c_[np.ones((data_size, 1)), x]
        self.coeff = np.linalg.inv(xm.T.dot(xm)).dot(xm.T).dot(y)
        self.fitted = True

    def regress(self, test_x):
        if not self.fitted:
            raise RegressionError("Fit the data before any predictions")
        mat_x = np.matrix(test_x).T
        data_size = mat_x.shape[0]
        mat_x = np.c_[np.ones((data_size, 1)), mat_x]

        return mat_x
