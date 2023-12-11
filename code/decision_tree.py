import numpy as np
from tqdm import tqdm


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)  # counts = number of [0,1]
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def information_gain(self, X, y, feature):
        entropy_before = self.entropy(y)
        unique_values = np.unique(X[feature])
        weighted_entropy_after = 0

        for value in unique_values:
            subset_indices = X[feature] == value
            subset_entropy = self.entropy(y[subset_indices])
            weighted_entropy_after += len(y[subset_indices]) / len(y) * subset_entropy

        return entropy_before - weighted_entropy_after  # Gain(S,A)

    def calculate_information_gain_for_all_features(self, X, y):
        information_gains = {}

        for feature in X.columns:
            information_gain = self.information_gain(X, y, feature)
            information_gains[feature] = information_gain

        return information_gains

    def find_best_split(self, X, y):
        best_information_gain = 0
        best_feature = None

        for feature in X.columns:
            current_information_gain = self.information_gain(X, y, feature)

            if current_information_gain > best_information_gain:
                best_information_gain = current_information_gain
                best_feature = feature

        return best_feature

    def build_tree(self, X, y, depth):
        if depth == 0 or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()  # most frequent class in y.

        best_feature = self.find_best_split(X, y)

        if best_feature is None:
            return np.bincount(y).argmax()

        tree = {best_feature: {}}

        for value in np.unique(X[best_feature]):
            subset_indices = X[best_feature] == value  # rows wich have best feature
            X_subset, y_subset = (
                X[subset_indices].drop(best_feature, axis=1),
                y[subset_indices],
            )
            tree[best_feature][value] = self.build_tree(X_subset, y_subset, depth - 1)

        return tree

    def fit(self, X, y):
        self.tree = self.build_tree(X, y, self.max_depth)

    def predict_single(self, x, tree, y):
        if not isinstance(tree, dict):
            return tree

        feature, subtree = next(iter(tree.items()))
        value = x.get(feature, None)

        if value is None or value not in subtree:
            return np.bincount(y).argmax()

        return self.predict_single(x, subtree[value], y)

    def predict(self, X, y):
        predictions = []

        for _, x in tqdm(X.iterrows(), total=len(X), desc="Predicting"):
            prediction = self.predict_single(x, self.tree, y)
            predictions.append(prediction)

        return predictions
