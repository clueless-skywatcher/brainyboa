from brainyboa.utils import Sigmoid
import numpy as np

if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5])
    s = Sigmoid(x)
    print(s.derivative(), s.evaluate())
