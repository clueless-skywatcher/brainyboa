import numpy as np

from sklearn.tree import DecisionTreeClassifier
from brainyboa.ensembling import CARTClassifier

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from brainyboa.metrics import mean_square_error

rs = np.random.RandomState(42)

iris = load_iris()
dec = DecisionTreeClassifier()
cart = CARTClassifier()
for _ in range(10):
    x_tr, x_t, y_tr, y_t = train_test_split(iris.data, iris.target, random_state = rs, test_size = 0.2)
    dec.fit(x_tr, y_tr)
    cart.fit(x_tr, y_tr)
    print(accuracy_score(y_t, dec.predict(x_t)), accuracy_score(y_t, cart.classify(x_t)))
