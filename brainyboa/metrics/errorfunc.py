import numpy as np

def squared_loss(x, y, theta = None):
    x1 = np.matrix(x)
    y1 = np.matrix(y)
    if theta is None:
        theta = np.zeros((1, x1.shape[0] + 1))
    y_pred = theta * x1.T
    return (np.sum(np.square(y_pred - y1))) / (2 * y1.shape[1])

def squared_loss_derivative(x, y, theta = None):
    x1 = np.matrix(x)
    y1 = np.matrix(y)
    if theta is None:
        theta = np.zeros((1, x1.shape[0] + 1))
    y_pred = theta * x1.T
    loss = y_pred - y1
    return (loss * x1) / y1.shape[1]

def mean_square_error(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.sum(np.square(x - y)) / len(x)
