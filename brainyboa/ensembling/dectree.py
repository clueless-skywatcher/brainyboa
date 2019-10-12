from ..metrics import shannon_entropy, gini_score
import operator
import numpy as np

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

class CARTClassifier:
    '''
    A decision tree classifier using CART Algorithm
    '''
    def __init__(self, metric = 'gini_score', min_sample_split = 2):
        self.root = None
        self.metric = eval(metric)
        self.min_sample_split = min_sample_split

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.data = np.column_stack((X, y))
        self._build_tree(self.data)
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
        curr_entropy = self.metric(data[:, -1])
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

        self.root = DecisionNode(best_feature, best_val, true_child, false_child)
        return self.root

    def _classify(self, x, node):
        if isinstance(node, PredictionLeaf):
            return node.predictions

        if isinstance(x[node.feature], int) or isinstance(x[node.feature], float):
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
