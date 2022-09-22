import argparse
import os.path
import warnings

from typing import Optional, List, Tuple

import numpy as np
from sting.classifier import Classifier
from sting.data import Feature, FeatureType, parse_c45

import util

# In Python, the convention for class names is CamelCase, just like Java! However, the convention for method and
# variable names is lowercase_separated_by_underscores, unlike Java.

class TreeNode():
    def __init__(self) -> None:
        self.children = []

        # Test Identifiers
        # If the test is on a nominal attribute, 'nominal_values' gives the node a way to remember
        # which nominal value (e.g. 'red') is associated with which child.
        self.attribute_index = None
        self.threshold = None
        self.nominal_values = []

        # If leaf_node property is true, this node will be given a label
        self.label = None

    @property
    def leaf_node(self):
        return (self.children == [])

class DecisionTree(Classifier):
    def __init__(self, schema: List[Feature], tree_depth_limit=0, use_information_gain=True):
        """
        This is the class where you will implement your decision tree. At the moment, we have provided some dummy code
        where this is simply a majority classifier in order to give you an idea of how the interface works. Don't forget
        to use all the good programming skills you learned in 132 and utilize numpy optimizations wherever possible.
        Good luck!
        """

        self._schema = schema  # For some models (like a decision tree) it makes sense to keep track of the data schema
        self._majority_label = 0  # Protected attributes in Python have an underscore prefix
        self.tree_depth_limit = tree_depth_limit
        self.use_information_gain = use_information_gain
        self.root = TreeNode()
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

    def _build_tree(self, X: np.ndarray, y: np.ndarray, possible_features: List, current_node: TreeNode, depth: int) -> TreeNode:
        """
        Recursive method for considering partitions and building the tree.

        Args:
            X: The dataset. The shape is (n_examples, n_features).
            y: The labels. The shape is (n_examples,)
            possible_features: The list of the indices of the schema's features that are left to choose from at this node.
            current_node: The TreeNode to be considered for a partition.
        """
        if self._pure_node(y):
            print("PURE NODE")
        if possible_features == []:
            print("NO MORE FEATURES")
        if depth == self.tree_depth_limit:
            print("I'M AT MY LIMIT")

        print(f"Shape: {np.shape(X)}")
        print(y)

        if self._pure_node(y) or possible_features == [] or depth == self.tree_depth_limit:
            self._make_majority_classifier(y, current_node)
        else: # prepare to partition:
            best_feature_index, best_feature_threshold = self._determine_split_criterion(X, y, possible_features)
            if best_feature_index == None: # ==> Max IG(X) = 0
                print("NO GOOD")
                self._make_majority_classifier(y, current_node)
            else:
                # remove feature from possible_features
                possible_features_updated = possible_features[:best_feature_index] + possible_features[best_feature_index+1:]
                current_node.attribute_index = best_feature_index
                # create children and partition
                if self.schema[best_feature_index].ftype == FeatureType.CONTINUOUS:
                    # Continuous Partition procedure:
                    print("Committing continuous node with attribute " + self.schema[best_feature_index].name + " at depth " + str(depth))
                    current_node.threshold = best_feature_threshold
                    child_one, child_two = TreeNode(), TreeNode()
                    self.size += 2
                    current_node.children.extend([child_one, child_two])
                    X_partition_leq, Y_partition_leq = self._partition_continuous(X, y, best_feature_index, best_feature_threshold, leq=True)
                    self._build_tree(X_partition_leq, Y_partition_leq, possible_features_updated, child_one, depth+1)
                    X_partition_g, Y_partition_g = self._partition_continuous(X, y, best_feature_index, best_feature_threshold, leq=False)
                    self._build_tree(X_partition_g, Y_partition_g, possible_features_updated, child_two, depth+1)
                
                elif self.schema[best_feature_index].ftype == FeatureType.NOMINAL:
                    # Nominal Partition procedure:
                    print("Committing nominal node with attribute " + self.schema[best_feature_index].name + " at depth " + str(depth))
                    for value in self.schema[best_feature_index].values:
                        child = TreeNode()
                        self.size += 1
                        current_node.nominal_values.append(value)
                        current_node.children.append(child)
                        X_partition, Y_partition = self._partition_nominal(X, y, best_feature_index, value)
                        self._build_tree(X_partition, Y_partition, possible_features_updated, child, depth+1)

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
        norm = np.linalg.norm(y, ord=1)
        size = np.size(y)
        if norm == size or norm == 0:
            return True # then node is pure
        else:
            return False

    def _partition_nominal(self, X: np.ndarray, y: np.ndarray, feature_index, value) -> Tuple[np.ndarray, np.ndarray]:
        """
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
        This is the method where the decision tree is evaluated.

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

    def _determine_split_criterion(self, X: np.ndarray, y: np.ndarray, possible_features: np.ndarray) -> Tuple[int, float]:
        """
        Determine decision tree split criterion. This is just an example to encourage you to use helper methods.
        Implement this however you like!
        """
        max_information_measure = 0
        best_feature_index = None
        best_threshold = None

        H_y = util.entropy(y)

        for index in possible_features:
            feature = self._schema[index]
            if feature.ftype == FeatureType.CONTINUOUS:
                # helper function that finds all possible thresholds
                dividers = self._find_dividers_jank(X[:, index], y)
                
                b = 0

                for div in dividers:
                    if self.use_information_gain:
                        current_IG = H_y - util.conditional_entropy(X, y, index, div)
                        if current_IG > max_information_measure:
                            max_information_measure = current_IG
                            best_feature_index = index
                            best_threshold = div
                        if current_IG > b:
                            b = current_IG
                    else:
                        current_GR = util.gain_ratio(X, y, index, div)
                        if current_GR > max_information_measure:
                            max_information_measure = current_GR
                            best_feature_index = index
                            best_threshold = div
                print(f"Checking continuous attribute {feature.name} ({index})...    IG = {b}")

            else:
                if self.use_information_gain:
                    current_IG = H_y - util.conditional_entropy(X, y, index, None)
                    print(f"Checking nominal attribute {feature.name} ({index})...    IG = {current_IG}")
                    if current_IG > max_information_measure:
                        max_information_measure = current_IG
                        best_feature_index = index
                else:
                    current_GR = util.gain_ratio(X, y, index, None)
                    if current_GR > max_information_measure:
                        max_information_measure = current_GR
                        best_feature_index = index
        
        return best_feature_index, best_threshold


    def _find_dividers_jank(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.unique(values)


    def _find_dividers(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
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
    You will implement this method.
    Given a trained decision tree and labelled dataset, Evaluate the tree and print metrics.
    """

    y_hat = dtree.predict(X)
    acc = util.accuracy(y, y_hat)
    print(f'Accuracy:{acc:.2f}')
    print('Size:', dtree.size)
    print('Maximum Depth:', dtree.tree_depth_limit)
    print('First Feature:', dtree.schema[0])


def dtree(data_path: str, tree_depth_limit: int, use_cross_validation: bool = True, information_gain: bool = True):
    """
    It is highly recommended that you make a function like this to run your program so that you are able to run it
    easily from a Jupyter notebook. This function has been PARTIALLY implemented for you, but not completely!

    :param data_path: The path to the data.
    :param tree_depth_limit: Depth limit of the decision tree
    :param use_cross_validation: If True, use cross validation. Otherwise, run on the full dataset.
    :param information_gain: If true, use information gain as the split criterion. Otherwise use gain ratio.
    :return:
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
    THIS IS YOUR MAIN FUNCTION. You will implement the evaluation of the program here. We have provided argparse code
    for you for this assignment, but in the future you may be responsible for doing this yourself.
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
