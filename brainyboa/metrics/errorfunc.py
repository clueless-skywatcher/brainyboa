import numpy as np
def mean_square_error(x, y):
    diff = np.array(x) - np.array(y)
    diff *= diff
    return np.sum(diff) / len(diff)

def root_mean_square_error(x, y):
    return np.sqrt(mean_square_error(x, y))

if __name__ == '__main__':
    print(root_mean_square_error([1,2,3], [4,5,6]))
