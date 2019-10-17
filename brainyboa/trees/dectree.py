from ..metrics import shannon_entropy, gini_score
import operator
import numpy as np
from brainyboa.base import *

__all__ = [
    'CARTClassifier',
    'CARTRegressor'
]

class DecisionNode:
    def __init__(self, feature, val, true_child = None, false_child = None):
        self.feature = feature
        if isinstance(val, float):
            self.val = int(val)
        else:
            self.val = val

        self.true_child = true_child
        self.false_child = false_child

class PredictionLeaf:
    def __init__(self, data):
        self.predictions = None
        counts = {}
        for row in data:
            label = row[-1]
            if isinstance(label, float):
                label = int(label)
            counts[label] = counts.get(label, 0) + 1
        self.predictions = counts

class RegressionLeaf:
    def __init__(self, rows):
        self.rows = np.array(rows)
        self.avg = np.mean(self.rows[:, -1])

class CARTClassifier(BaseClassifier):
    '''
    A decision tree classifier using CART Algorithm
    '''
    def __init__(self, metric = 'gini_score', min_sample_split = 2):
        self.root = None
        self.metric = eval(metric)
        self.min_sample_split = min_sample_split
        super().__init__()

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.data = np.column_stack((X, y))
        self.root = self._build_tree(self.data)
        return self

    def _split(self, data, feature, value):
        true_rows, false_rows = [], []
        for row in data:
            if isinstance(row[feature], int) or isinstance(row[feature], float):
                if row[feature] >= value:
                    true_rows.append(row)
                else:
                    false_rows.append(row)
            else:
                if row[feature] == value:
                    true_rows.append(row)
                else:
                    false_rows.append(row)
        return np.array(true_rows), np.array(false_rows)

    def _info_gain(self, true, false, entropy):
        p = float(len(true)) / float(len(true) + len(false))
        return entropy - (p * self.metric(true[:, -1]) + (1 - p) * self.metric(false[:, -1]))

    def _best_split_condition(self, data):
        best_gain = 0.0
        best_feature = 0
        best_val = None
        curr_entropy = 1
        columns = data.shape[1] - 1
        for feature in range(columns):
            vals = set(row[feature] for row in data)
            for val in vals:
                true_rows, false_rows = self._split(data, feature, val)
                if len(true_rows) < self.min_sample_split or len(false_rows) < self.min_sample_split:
                    continue
                gain = self._info_gain(true_rows, false_rows, curr_entropy)
                if gain > best_gain:
                    best_gain, best_feature, best_val = gain, feature, val

        return best_gain, best_feature, best_val

    def _build_tree(self, data):
        best_gain, best_feature, best_val = self._best_split_condition(data)

        if best_gain == 0:
            return PredictionLeaf(data)

        true, false = self._split(data, best_feature, best_val)

        true_child = self._build_tree(true)
        false_child = self._build_tree(false)

        root = DecisionNode(best_feature, best_val, true_child, false_child)
        return root

    def _classify(self, x, node):
        if isinstance(node, PredictionLeaf):
            return node.predictions

        elif isinstance(x[node.feature], int) or isinstance(x[node.feature], float):
            if x[node.feature] >= node.val:
                return self._classify(x, node.true_child)
            else:
                return self._classify(x, node.false_child)
        else:
            if x[node.feature] == node.val:
                return self._classify(x, node.true_child)
            else:
                return self._classify(x, node.false_child)

    def classify(self, x):
        preds = []
        for row in x:
            pred = self._classify(row, self.root)
            key = max(pred.items(), key = operator.itemgetter(1))[0]
            if isinstance(key, float):
                key = int(key)
            preds.append(key)
        return np.array(preds)

class RegressionTree:
    def __init__(self, feature, val, true_child = None, false_child = None):
        self.feature = feature
        self.true_child = true_child
        self.false_child = false_child
        self.val = val


class CARTRegressor(BaseRegressor):
    def __init__(self, min_sample_split = 6, tolerance = 0.5):
        self.fitted = False
        self.root = None
        self.min_sample_split = min_sample_split
        self.tolerance = tolerance

    def _split(self, data, feature, value):
        true_rows = np.array(data[np.nonzero(data[:, feature] > value)[0], :])
        false_rows = np.array(data[np.nonzero(data[:, feature] <= value)[0], :])
        return true_rows, false_rows

    def fit(self, X, y):
        self.data = np.column_stack((X, y))
        self.root = self._build_tree(self.data)
        return self

    def _build_tree(self, data):
        feat, val = self._best_split_condition(data)
        if feat == None:
            return val

        tree = RegressionTree(feat, val)
        true_rows, false_rows = self._split(data, feat, val)
        tree.true_child = self._build_tree(true_rows)
        tree.false_child = self._build_tree(false_rows)

        return tree

    def _get_error(self, dataset):
        return np.var(dataset[:, -1]) * np.shape(dataset)[0]

    def _best_split_condition(self, data):
        if len(np.unique(data[:, -1])) == 1:
            return None, np.mean(data[:, -1])
        rows, cols = np.shape(data)
        err = self._get_error(data)
        min_err, best_feature, best_val = np.inf, 0, 0
        for feature in range(cols - 1):
            for val in set(data[:, feature]):
                true_rows, false_rows = self._split(data, feature, val)
                if len(true_rows) < self.min_sample_split or len(false_rows) < self.min_sample_split:
                    continue
                new_err = self._get_error(true_rows) + self._get_error(false_rows)
                if new_err < min_err:
                    best_feature = feature
                    best_val = val
                    min_err = new_err
        if (err - min_err) < self.tolerance:
            return None, np.mean(data[:, -1])
        true_rows, false_rows = self._split(data, best_feature, best_val)
        if len(true_rows) < self.min_sample_split or len(false_rows) < self.min_sample_split:
            return None, np.mean(data[:, -1])
        return best_feature, best_val

    def _regress(self, x, node):
        if not isinstance(node, RegressionTree):
            return node
        if x[node.feature] >= node.val:
            return self._regress(x, node.true_child)
        else:
            return self._regress(x, node.false_child)

    def regress(self, X):
        X = np.array(X)
        preds = []
        for x in X:
            pred = self._regress(x, self.root)
            preds.append(pred)
        return preds
