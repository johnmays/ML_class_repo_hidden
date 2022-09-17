import argparse
import os.path
import warnings

from typing import Optional, List

import numpy as np
from sting.classifier import Classifier
from sting.data import Feature, parse_c45

import util

# In Python, the convention for class names is CamelCase, just like Java! However, the convention for method and
# variable names is lowercase_separated_by_underscores, unlike Java.

class TreeNode():
    def __init__(self) -> None:
        self.children = []
        self.attribute = None
        # jkm100 -- probably going to need some way to identify the test <-- here (index, threshold)
        self.leaf_node = True
        self.partition = None
        self.label = None # jkm100 -- going to need to make this 0,1 at some point - only needed for leaf nodes

    @property
    def leaf_node(self):
        if self.children == []:
            self.leaf_node = True
        else:
            self.leaf_node = False
        return self.leaf_node

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

    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        """
        This is the method where the training algorithm will run.

        Args:
            X: The dataset. The shape is (n_examples, n_features).
            y: The labels. The shape is (n_examples,)
            weights: Weights for each example. Will become relevant later in the course, ignore for now.
        """
        # jkm100 -- need to add condition here: no features? if yes, then make majority classifier w/ TreeNode.label attribute
        self._build_tree(X, y, np.copy(self.schema), self.root)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, possible_features, current_node):
        if possible_features == [] or self._pure_node(y):
            pass # leave node as is --> leaf node
        else: # prepare to partition:
            best_feature_index = self._determine_split_criterion
            if best_feature_index == None: # ==> Max IG(X) = 0
                # jkm100 -- need to add: if yes, then make majority classifier w/ TreeNode.label attribute
                pass
            else:
                # remove feature from possible_features
                # create children
                # call on them
                # REMAINS TO BE IMPLEMENTED
                pass

    def _pure_node(self, y: np.ndarray):
        norm = np.linalg.norm(y, ord=1)
        size = np.size(y)
        if norm == size or norm == 0:
            return True # then node is pure
        else:
            return False


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This is the method where the decision tree is evaluated.

        Args:
            X: The testing data of shape (n_examples, n_features).

        Returns: Predictions of shape (n_examples,), either 0 or 1
        """
        raise NotImplementedError()

    @property
    def schema(self):
        """
        Returns: The dataset schema
        """
        return self._schema

    # It is standard practice to prepend helper methods with an underscore "_" to mark them as protected.
    def _determine_split_criterion(self, X: np.ndarray, y: np.ndarray) -> int:
        """
        Determine decision tree split criterion. This is just an example to encourage you to use helper methods.
        Implement this however you like!
        """
        max_information_measure = 0
        best_feature_index = None
        for index in range(possible_features.shape()[0]):
            if possible_features[index].ftype == Feature.FeatureType.CONTINUOUS:
                # helper function that finds all possible thresholds
                # pass over all possible thresholds
                # NOT YET IMPLEMENTED
                pass
            else:
                if self.use_information_gain:
                    current_IG = util.information_gain(X, y, index, None)
                    if current_IG > max_information_measure:
                        max_information_measure = current_IG
                        best_feature_index = index
                else:
                    current_GR = util.gain_ratio(X, y, index, None)
                    if current_GR > max_information_measure:
                        max_information_measure = current_GR
                        best_feature_index = index
        return best_feature_index


def evaluate_and_print_metrics(dtree: DecisionTree, X: np.ndarray, y: np.ndarray):
    """
    You will implement this method.
    Given a trained decision tree and labelled dataset, Evaluate the tree and print metrics.
    """

    y_hat = dtree.predict(X)
    acc = util.accuracy(y, y_hat)
    print(f'Accuracy:{acc:.2f}')
    print('Size:', 0)
    print('Maximum Depth:', dtree.tree_depth_limit)
    print('First Feature:', dtree.schema[0])

    raise NotImplementedError()


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
