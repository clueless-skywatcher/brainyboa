from brainyboa.ensembling import CARTClassifier, print_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

if __name__ == '__main__':
    cart = CARTClassifier()
    dec = DecisionTreeClassifier()
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.2)
    cart.fit(x_train, y_train)
    dec.fit(x_train, y_train)
    print(accuracy_score(dec.predict(x_test), y_test))
    print(accuracy_score(cart.classify(x_test), y_test))
