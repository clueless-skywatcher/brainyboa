from brainyboa.linear import LinearRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from brainyboa.linear import root_mean_square_error
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    lin = LinearRegressor()
    lin2 = LinearRegression()
    X_t_1 = np.random.rand(10, 1) * 5
    X_t_2 = np.random.rand(10, 1) * 5
    X_t = np.concatenate((X_t_1, X_t_2), axis = 1)
    Y_t = 6 * X_t_1 + 5 * X_t_2 + np.random.rand(10, 1) + 8
    lin.fit(X_t, Y_t)
    lin2.fit(X_t, Y_t)
    X_t_1 = np.random.rand(10, 1) * 5
    X_t_2 = np.random.rand(10, 1) * 5
    X_t = np.concatenate((X_t_1, X_t_2), axis = 1)
    Y_t = 6 * X_t_1 + 5 * X_t_2 + np.random.rand(10, 1) + 8
    pred1 = lin.regress(X_t)
    print(root_mean_square_error(pred1, Y_t))
    pred1 = lin2.predict(X_t)
    print(root_mean_square_error(pred1, Y_t))
