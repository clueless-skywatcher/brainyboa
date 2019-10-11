import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from brainyboa.linear import RidgeRegressor, LinearRegressor
from brainyboa.graddesc import GradientDescentRegressor

X = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[1], [2]])

ridge1 = LinearRegression()
ridge1.fit(X, y)
ridge2 = LinearRegressor()
ridge2.fit(X, y)

print(ridge1.predict([[1, 2, 6], [1, 19, 8]]))
print(ridge2.regress([[1, 2, 6], [1, 19, 8]]))
