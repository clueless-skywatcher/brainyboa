import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import make_classification, make_regression
from brainyboa.trees import CARTRegressor, CARTClassifier
from brainyboa.linear import *
from brainyboa.metrics import acc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import *

for _ in range(1):
    X, y = make_regression(n_samples = 100, n_features = 6)
    cart = CARTRegressor()
    dec = DecisionTreeRegressor()
    x_tr, x_t, y_tr, y_t = train_test_split(X, y, test_size = 0.2)
    cart.fit(x_tr, y_tr)
    dec.fit(x_tr, y_tr)
    preds1 = cart.regress(x_t)
    preds2 = dec.predict(x_t)
    for i in range(len(x_t)):
        print(f"Prediction1 = {preds1[i]}, Prediction2 = {preds2[i]}, Actual = {y_t[i]}")
