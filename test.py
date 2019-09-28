from brainyboa.metrics import *
from brainyboa.linear import LinearRegressor
from brainyboa.graddesc import GradientDescentRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    X, y = make_regression(n_samples = 10, n_features = 1, noise = 0.9)
    gd = GradientDescentRegressor()
    gd.fit(X, y)
    print(gd.regress([[2], [4], [1]]))
    sgd = SGDRegressor(max_iter = 15000, alpha = 0.01)
    sgd.fit(X, y)
    print(sgd.predict([[2], [4], [1]]))
    lin = LinearRegression()
    lin.fit(X, y)
    print(lin.predict([[2], [4], [1]]))
