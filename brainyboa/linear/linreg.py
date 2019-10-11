from brainyboa.metrics import mean_square_error, squared_loss, squared_loss_derivative
import numpy as np
from brainyboa.linear.exc import RegressionError

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
        self.ms = np.linalg.pinv(xm.T.dot(xm)).dot(xm.T).dot(y)
        self.fitted = True
        self.coeffs = np.asarray(self.ms[1:]).reshape(-1)
        self.intercept = np.asscalar(self.ms[0])
        self.x = x
        self.y = y
        return self

    def regress(self, test_x):
        if not self.fitted:
            raise RegressionError("Fit the data before any predictions")
        mat_x = np.matrix(test_x)
        data_size = mat_x.shape[0]
        mat_x = np.c_[np.ones((data_size, 1)), mat_x]
        pred = mat_x.dot(self.ms)
        return pred

class RidgeRegressor:
    def __init__(self, alpha = 1):
        self.alpha = alpha

    def fit(self, data_x, data_y):
        x = np.matrix(data_x)
        y = np.matrix(data_y)
        if y.shape[0] == 1:
            y = y.T
        data_size, feature_size = x.shape
        xm = np.c_[np.ones((data_size, 1)), x]
        A = np.identity(xm.shape[1])
        self.ms = np.linalg.inv((xm.T.dot(xm)) + self.alpha * A).dot(xm.T).dot(y)
        self.fitted = True
        self.coeffs = np.asarray(self.ms[1:]).reshape(-1)
        self.intercept = np.asscalar(self.ms[0])
        self.x = x
        self.y = y
        return self

    def regress(self, test_x):
        if not self.fitted:
            raise RegressionError("Fit the data before any predictions")
        mat_x = np.matrix(test_x)
        data_size = mat_x.shape[0]
        mat_x = np.c_[np.ones((data_size, 1)), mat_x]
        pred = mat_x.dot(self.ms)
        return pred
