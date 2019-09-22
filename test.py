from brainyboa.linear import LinearRegressor
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    lin = LinearRegressor()
    X = np.random.rand(100, 1)
    Y = 4 + 3*X + np.random.rand(100, 1)
    lin.fit(X, Y)
    print(lin.predict([[0, 2, 5]]))
