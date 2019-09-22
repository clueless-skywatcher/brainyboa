from brainyboa.linear import LinearRegressor
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    lin = LinearRegressor()
    X1 = np.random.rand(10, 1) * 5
    X2 = np.random.rand(10, 1) * 5
    Y = 4 * X1 + 5 * X2 + np.random.rand(10, 1) + 8
    X = np.concatenate((X1, X2), axis = 1)
    lin.fit(X, Y)
    X_t = np.matrix([[1, 2], [3, 4], [4, 5]])
    pred1 = lin.regress(X_t)
    print(pred1)
