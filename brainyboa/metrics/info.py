import numpy as np

__all__ = [
    'shannon_entropy',
    'gini_score'
]

def shannon_entropy(array):
    '''
    Calculates the Shannon Entropy of a given 1D array

    :param array:
    '''
    array = np.array(array)
    counts = {}
    for x in array:
        counts[x] = counts.get(x, 0) + 1
    entropy = 0.0
    for x in counts:
        prob = counts[x] / len(array)
        entropy -= prob * np.log2(prob)
    return entropy

def gini_score(array):
    '''
    Calculates the Gini impurity of a given 1D array

    :param array:
    '''
    array = np.array(array)
    counts = {}
    for x in array:
        counts[x] = counts.get(x, 0) + 1
    score = 0.0
    for x in counts:
        prob = counts[x] / len(array)
        score += prob ** 2
    return 1 - score
