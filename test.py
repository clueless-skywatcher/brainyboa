import numpy as np
from sklearn.tree import DecisionTreeClassifier
from brainyboa.ensembling import CARTClassifier
from sklearn.datasets import load_iris

iris = load_iris()
dec = DecisionTreeClassifier()
cart = CARTClassifier()
dec.fit(iris.data, iris.target)
cart.fit(iris.data, iris.target)

print(dec.decision_path([iris.data[0]]))
