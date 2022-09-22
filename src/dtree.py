import argparse
from locale import YESEXPR
import os.path
import warnings

from typing import Optional, List, Tuple

import numpy as np
from sting.classifier import Classifier
from sting.data import Feature, FeatureType, parse_c45

import util

class TreeNode():
    def __init__(self) -> None:
        """
        This is the data structure for the tree that will make the classifier.
        It contains information about the test it runs, its children, and if it is a leaf node, the label it will give to examples.
        """
        # Test Identifiers:
        self.attribute_index = None
        self.threshold = None
        self.nominal_values = [] # Should populate in the same order as self.children

        # will be set to 0 or 1 if self.leaf_node is true
        self.label = None

        self.children = []

    @property
    def leaf_node(self):
        return (self.children == [])

class DecisionTree(Classifier):
    def __init__(self, schema: List[Feature], tree_depth_limit=0, use_information_gain=True):
        """
        This class handles the DecisionTree.  It has methods for building the tree (fit()), and a method for predicting on new examples.
        """
        self._schema = schema  # A set of features with their properties
        self.tree_depth_limit = tree_depth_limit
        self.use_information_gain = use_information_gain
        self.root = TreeNode() # beginning with a default TreeNode as the root
        self.size = 1

    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        """
        This is the method where the training algorithm will run.

        Args:
            X: The dataset. The shape is (n_examples, n_features).
            y: The labels. The shape is (n_examples,)
            weights: Weights for each example. Will become relevant later in the course, ignore for now.
        """
        print(f"FITTING {len(X)} EXAMPLES WITH {len(self._schema)} ATTRIBUTES")
        if self.schema == []:
            self._make_majority_classifier(y, self.root)
        else:
            self._build_tree(X, y, list(range(len(self.schema))), self.root, 0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, possible_attributes: List, current_node: TreeNode, depth: int) -> TreeNode:
        """
        Recursive method for considering partitions and building the tree.
        Made after the ID3 algorithm.

        Args:
            X: The dataset. The shape is (n_examples, n_features).
            y: The labels. The shape is (n_examples,)
            possible_attributes: The list of the indices of the schema's features that are left to choose from at this node.
            current_node: The TreeNode to be considered for a partition.
            depth: a measure of how deep current_node is in the tree
        """

        print(f"Shape: {np.shape(X)}")
        print(y)

        if self._pure_node(y) or possible_attributes == [] or depth == self.tree_depth_limit:
            self._make_majority_classifier(y, current_node)
        else: # prepare to partition:
            best_attribute_index, best_attribute_threshold = self._determine_split_criterion(X, y, possible_attributes)
            if best_attribute_index == None: # ==> Max IG(X) = 0
                self._make_majority_classifier(y, current_node)
            else:
                # remove feature from possible_attributes
                possible_attributes_updated = possible_attributes[:best_attribute_index] + possible_attributes[best_attribute_index+1:]
                current_node.attribute_index = best_attribute_index
                # create children and partition
                if self.schema[best_attribute_index].ftype == FeatureType.CONTINUOUS:
                    # Continuous Partition procedure:
                    print("Committing continuous node with attribute " + self.schema[best_attribute_index].name + ", value " + str(best_attribute_threshold) + " at depth " + str(depth))

                    current_node.threshold = best_attribute_threshold

                    child_one, child_two = TreeNode(), TreeNode()
                    current_node.children.extend([child_one, child_two])
                    self.size += 2
                    
                    # Partitioning for less than or equal to the threshold
                    X_partition_leq, Y_partition_leq = self._partition_continuous(X, y, best_attribute_index, best_attribute_threshold, leq=True)
                    self._build_tree(X_partition_leq, Y_partition_leq, possible_attributes_updated, child_one, depth+1)
                    # Partitioning for greater than the threshold
                    X_partition_g, Y_partition_g = self._partition_continuous(X, y, best_attribute_index, best_attribute_threshold, leq=False)
                    self._build_tree(X_partition_g, Y_partition_g, possible_attributes_updated, child_two, depth+1)
                else: 
                    # Nominal Partition procedure:
                    print("Committing nominal node with attribute " + self.schema[best_attribute_index].name + " at depth " + str(depth))
                    for value in self.schema[best_attribute_index].values:
                        child = TreeNode()
                        current_node.nominal_values.append(value)
                        current_node.children.append(child)
                        self.size += 1
                        
                        X_partition, Y_partition = self._partition_nominal(X, y, best_attribute_index, value)
                        self._build_tree(X_partition, Y_partition, possible_attributes_updated, child, depth+1)

    def _make_majority_classifier(self, y: np.ndarray, node: TreeNode) -> TreeNode:
        """
        Helper method for turning a node into a majority classifier.

        Args:
            y: The labels. The shape is (n_examples,)
            node: The TreeNode that will be turned into a simple majority classifier.
        """
        n_zero, n_one = util.count_label_occurrences(y)
        if n_one > n_zero:
            node.label = 1
        else:
            node.label = 0
        return node

    def _pure_node(self, y: np.ndarray):
        """
        Helper method for determining if a node is pure.

        Args:
            y: The labels. The shape is (n_examples,)

        Returns: boolean that is true if (every label in y is 1) or (every label in y is 0)
        """
        n_zeros, n_ones = util.count_label_occurrences(y)
        size = np.size(y)
        if n_zeros == size or n_ones == 0: # then node is pure
            return True
        else:
            return False

    def _partition_nominal(self, X: np.ndarray, y: np.ndarray, feature_index, value) -> Tuple[np.ndarray, np.ndarray]:
        """
        A method for partitioning a subset of training data

        Args:
            X: A subset of the examples in the dataset that the returned partition will be nondestructively taken from. Shape is (n_examples, n_features).
            y: A corresponding subset of the labels in the dataset that the returned partition will be nondestructively taken from. Shape is (n_examples, ).
            feature_index: The (row) index of X where X's measure of a specific nominal attribute lie.
            value: the value of the nominal attribute that positively determines partition memmbership,

        Returns: the subset of X and y corresponding to the partition based upon the nominal value
        """
        X_partitioned = []
        y_partitioned = []
        for example, label in zip(X,y):
            if example[feature_index] == value:
                X_partitioned.append(example)
                y_partitioned.append(label)
        return np.array(X_partitioned), np.array(y_partitioned)

    def _partition_continuous(self, X: np.ndarray, y: np.ndarray, feature_index, threshold, leq: bool):
        """
        A method for partitioning a subset of training data

        Args:
            X: A subset of the examples in the dataset that the returned partition will be nondestructively taken from. Shape is (n_examples, n_features).
            y: A corresponding subset of the labels in the dataset that the returned partition will be nondestructively taken from. Shape is (n_examples, ).
            feature_index: The (row) index of X where X's measure of a specific nominal attribute lie.
            threshold: the value of the continous attribute that X & y will be partitioned by.
            leq: True  ==> partition will contain all examples less than or equal to the threshold.
                False ==> partition will contain all examples greater than the threshold.

        Returns: the subset of X and y corresponding to the partition either (greater than) or (less than or equal to) the threshold
        """
        X_partitioned = []
        y_partitioned = []
        for example, label in zip(X,y):
            if leq and example[feature_index] <= threshold:
                X_partitioned.append(example)
                y_partitioned.append(label)
            elif not leq and example[feature_index] > threshold:
                X_partitioned.append(example)
                y_partitioned.append(label)
        return np.array(X_partitioned), np.array(y_partitioned)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method uses the decision tree to label a set of examples, X.

        Args:
            X: The testing data of shape (n_examples, n_features).

        Returns: Predictions of shape (n_examples,), either 0 or 1
        """
        labels = []
        for example in X:
            current_node = self.root
            while not current_node.leaf_node:

                # Get the test at the current node
                index = current_node.attribute_index
                threshold = current_node.threshold
                
                # Set current_node to the appropriate child of the current_node
                if threshold is None:
                    for i, nv in enumerate(current_node.nominal_values):
                        if nv == example[index]:
                            current_node = current_node.children[i]
                            break
                else:
                    if example[index] <= threshold:
                        current_node = current_node.children[0]
                    else:
                        current_node = current_node.children[1]
            
            labels.append(current_node.label)

        return np.array(labels)     

    @property
    def schema(self):
        """
        Returns: The dataset schema
        """
        return self._schema

    def _determine_split_criterion(self, X: np.ndarray, y: np.ndarray, possible_attributes: np.ndarray) -> Tuple[int, float]:
        """
        Args:
            X: The dataset. The shape is (n_examples, n_features).
            y: The labels. The shape is (n_examples,)
            possible_attributes: The list of the indices of the schema's features that are left to choose from at this node.

        Returns: best_attribute_index, the schema index of the best possible attribute by the given metric, and best_threshold, a corresponding threshold value for that attribute if it is continuous (= None otherwise).
        """
        max_information_measure = 0
        best_attribute_index = None
        best_threshold = None

        H_y = util.entropy(y)

        for index in possible_attributes:
            feature = self._schema[index]
            if feature.ftype == FeatureType.CONTINUOUS:
                # helper function that finds all possible thresholds
                dividers = self._find_thresholds(X[:, index], y)
                
                b = 0

                for div in dividers:
                    if self.use_information_gain:
                        current_IG = H_y - util.conditional_entropy(X, y, index, div)
                        if current_IG > max_information_measure:
                            max_information_measure = current_IG
                            best_attribute_index = index
                            best_threshold = div
                        if current_IG > b:
                            b = current_IG
                    else:
                        current_GR = (H_y - util.conditional_entropy(X, y, index, div)) / util.attribute_entropy(X, index, div)
                        if current_GR > max_information_measure:
                            max_information_measure = current_GR
                            best_attribute_index = index
                            best_threshold = div
                        if current_GR > b:
                            b = current_GR
                
                if self.use_information_gain:
                    print(f"Checking continuous attribute {feature.name} ({index})...    IG = {b}")
                else:
                    print(f"Checking continuous attribute {feature.name} ({index})...    GR = {b}")

            else:
                if self.use_information_gain:
                    current_IG = H_y - util.conditional_entropy(X, y, index, None)
                    print(f"Checking nominal attribute {feature.name} ({index})...    IG = {current_IG}")
                    if current_IG > max_information_measure:
                        max_information_measure = current_IG
                        best_attribute_index = index
                else:
                    current_GR = (H_y - util.conditional_entropy(X, y, index, None)) / util.attribute_entropy(X, index, None)
                    print(f"Checking nominal attribute {feature.name} ({index})...    GR = {current_GR}")
                    if current_GR > max_information_measure:
                        max_information_measure = current_GR
                        best_attribute_index = index
        
        return best_attribute_index, best_threshold

    def _find_thresholds(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        NOT YET DOCUMENTED
        This method returns the points for a continuous attribute that could be possible thresholds.
        """
        return np.unique(values)

    def _find_thresholds_2(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        NOT YET DOCUMENTED
        This method returns the points for a continuous attribute that could be possible thresholds, in a different way.
        """ 
        values_sort = [[v, l] for v, l in zip(values, labels)]
        values_sort = sorted(values_sort, key=lambda x: x[0])

        dividers = [values_sort[0][0] - 0.01]
        for i in range(1, len(labels)):
            if values_sort[i][1] != values_sort[i-1][1]:
                dividers.append((values_sort[i][0] + values_sort[i-1][0]) / 2)
        dividers.append(values_sort[-1][0] + 0.01)
        
        return np.array(dividers)


def evaluate_and_print_metrics(dtree: DecisionTree, X: np.ndarray, y: np.ndarray):
    """
    NOT YET DOCUMENTED
    """

    y_hat = dtree.predict(X)
    acc = util.accuracy(y, y_hat)
    print(f'Accuracy:{acc:.2f}')
    print('Size:', dtree.size)
    print('Maximum Depth:', dtree.tree_depth_limit)
    print('First Feature:', dtree.schema[0])


def dtree(data_path: str, tree_depth_limit: int, use_cross_validation: bool = True, information_gain: bool = True):
    """
    NOT YET DOCUMENTED

    Args: 
        data_path: The path to the data.
        tree_depth_limit: Depth limit of the decision tree
        use_cross_validation: If True, use cross validation. Otherwise, run on the full dataset.
        information_gain: If true, use information gain as the split criterion. Otherwise use gain ratio.
    """

    # last entry in the data_path is the file base (name of the dataset)
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
    root_dir = os.sep.join(path[:-1])
    schema, X, y = parse_c45(file_base, root_dir)

    # print(schema)

    if use_cross_validation:
        datasets = util.cv_split(X, y, folds=5, stratified=True)
    else:
        datasets = ((X, y, X, y),)

    for X_train, y_train, X_test, y_test in datasets:
        decision_tree = DecisionTree(schema,
            tree_depth_limit=tree_depth_limit,
            use_information_gain=information_gain)
        decision_tree.fit(X_train, y_train)
        evaluate_and_print_metrics(decision_tree, X_test, y_test)


if __name__ == '__main__':
    """
    NOT YET DOCUMENTED
    """

    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a decision tree algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('depth_limit', metavar='DEPTH', type=int,
                        help='Depth limit of the tree. Must be a non-negative integer. A value of 0 sets no limit.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.add_argument('--use-gain-ratio', dest='gain_ratio', action='store_true',
                        help='Use gain ratio as tree split criterion instead of information gain.')
    parser.set_defaults(cv=True, gain_ratio=False)
    args = parser.parse_args()

    # If the depth limit is negative throw an exception
    if args.depth_limit < 0:
        raise argparse.ArgumentTypeError('Tree depth limit must be non-negative.')

    # You can access args with the dot operator like so:
    data_path = os.path.expanduser(args.path)
    tree_depth_limit = args.depth_limit
    use_cross_validation = args.cv
    use_information_gain = not args.gain_ratio

    dtree(data_path, tree_depth_limit, use_cross_validation, use_information_gain)
