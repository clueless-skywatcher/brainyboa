import numpy as np

def acc_score(x, y):
    '''
    Calculates the accuracy score of x w.r.t y
    '''
    sum = 0
    for i in range(len(x)):
        if x[i] == y[i]:
            sum += 1
    return sum / len(x)

def rsq_score(y_true, y_pred):
    '''
    Calculates the R^2 score of y_pred w.r.t y_true. R^2 is a statistic that will give some
    information about the goodness of fit of a model. In regression, the R^2 coefficient of
    determination is a statistical measure of how well the regression predictions approximate the
    real data points. An R^2 of 1 indicates that the regression predictions perfectly fit the data.
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diff1 = ((y_true - y_pred) ** 2).sum()
    diff2 = ((y_true - y_true.mean()) ** 2).sum()
    if diff2 == 0:
        return 1.0
    return 1 - (diff1 / diff2)

def variance(x):
    x = np.array(x)
    mean = np.ones(np.shape(x)) * x.mean(axis = 0)
    n = np.shape(x)[0]
    var = (1 / n) * (np.square(x - mean))
    return var
