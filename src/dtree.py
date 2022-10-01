import argparse
import os.path
from queue import Empty
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
        self.tree_depth_limit = -1 if tree_depth_limit == 0 else tree_depth_limit
        self.use_information_gain = use_information_gain
        self.root = TreeNode() # beginning with a default TreeNode as the root
        self.size = 1
        self.depth = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, research:bool = False) -> None:
        # weights: Optional[np.ndarray] = None, 
        """
        This is the method where the training algorithm will run.

        Args:
            X: The dataset. The shape is (n_examples, n_features).
            y: The labels. The shape is (n_examples,)
            weights: Weights for each example. Will become relevant later in the course, ignore for now.
            research: will build the tree acoording to the research question instead of the normal way if true.
        """
        print(f"FITTING {len(X_train)} EXAMPLES WITH {len(self._schema)} ATTRIBUTES")
        if self.schema == []:
            self._make_majority_classifier(y_train, self.root)
        else:
            # remove any 'id' attributes before starting
            possible_attributes = [i for i in range(len(self.schema)) if self.schema[i].name[-2:] != "id"]
            if not research:
                self._build_tree(X_train, y_train, possible_attributes, self.root, 0)
            else:
                if X_val is None:
                    raise argparse.ArgumentError("fit: X_val was not described for some reason");
                self._build_tree_research(X_train, y_train, X_val, y_val, possible_attributes, self.root, 0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, possible_attributes: List, current_node: TreeNode, depth: int) -> TreeNode:
        """
        Recursive method for considering partitions and building the tree.
        Made after the ID3 algorithm.

        Args:
            X_train: The dataset. The shape is (n_examples, n_features).
            y_train: The labels. The shape is (n_examples,)
            possible_attributes: The list of the indices of the schema's features that are left to choose from at this node.
            current_node: The TreeNode to be considered for a partition.
            depth: a measure of how deep current_node is in the tree
        """
        print(f"CREATING NODE ON {len(X)} EXAMPLES (depth: {depth})", end='\r')
        
        if depth > self.depth:
            self.depth = depth

        if self._pure_node(y):
            print("** Pure node, skipping **", end='\r')
        elif possible_attributes == []:
            print("** No more features, skipping **", end='\r')
        elif depth == self.tree_depth_limit:
            print("** At depth limit, skipping **", end='\r')
        
        if self._pure_node(y) or possible_attributes == [] or depth == self.tree_depth_limit:
            self._make_majority_classifier(y, current_node)
        else: # prepare to partition:
            best_attribute_index, best_attribute_threshold = self._determine_split_criterion(X, y, possible_attributes)
            if best_attribute_index == None: # ==> Max IG(X) = 0
                self._make_majority_classifier(y, current_node)
            else:
                # create children and partition
                if self.schema[best_attribute_index].ftype == FeatureType.CONTINUOUS:
                    # Note: we do not update the possible attributes list here bc continuous tests may be made again on different thresholds.
                    # Continuous Partition procedure:
                    print("Assigning node continuous attribute " + self.schema[best_attribute_index].name + ", value " + str(best_attribute_threshold) + " at depth (" + str(depth) + ") ", end="\r")
                    current_node.attribute_index = best_attribute_index
                    current_node.threshold = best_attribute_threshold

                    child_one, child_two = TreeNode(), TreeNode()
                    current_node.children.extend([child_one, child_two])
                    self.size += 2
                    
                    # Partitioning for less than or equal to the threshold
                    X_partition_leq, Y_partition_leq = self._partition_continuous(X, y, best_attribute_index, best_attribute_threshold, leq=True)
                    self._build_tree(X_partition_leq, Y_partition_leq, possible_attributes, child_one, depth+1)
                    # Partitioning for greater than the threshold
                    X_partition_g, Y_partition_g = self._partition_continuous(X, y, best_attribute_index, best_attribute_threshold, leq=False)
                    self._build_tree(X_partition_g, Y_partition_g, possible_attributes, child_two, depth+1)
                else: 
                    # Nominal Partition procedure:
                    possible_attributes_updated = [i for i in possible_attributes if i != best_attribute_index] # (removed feature from possible_attributes)
                    current_node.attribute_index = best_attribute_index
                    print("Assigning node nominal attribute " + self.schema[best_attribute_index].name + " at depth (" + str(depth) + ")", end="\r")
                    for value in self.schema[best_attribute_index].values:
                        child = TreeNode()
                        current_node.nominal_values.append(value)
                        current_node.children.append(child)
                        self.size += 1
                        
                        X_partition, Y_partition = self._partition_nominal(X, y, best_attribute_index, value)
                        self._build_tree(X_partition, Y_partition, possible_attributes_updated, child, depth+1)

    def _build_tree_research(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, possible_attributes: List, current_node: TreeNode, depth: int, temp_depth:bool = None) -> TreeNode:
        """
        (Research Extension version)
        Recursive method for considering partitions and building the tree.
        Augmented ID3 algorithm.

        Args:
            X_train: The training dataset. The shape is (n_examples, n_features).
            y_train: The training labels. The shape is (n_examples,)
            X_val: The validation dataset. The shape is (n_examples, n_features).
            y_val: The validation labels. The shape is (n_examples,)
            possible_attributes: The list of the indices of the schema's features that are left to choose from at this node.
            current_node: The TreeNode to be considered for a partition.
            depth: a measure of how deep current_node is in the tree
            temp_depth: an argument to determine if a temporary tree is being built, and how far to take it.
        """
        print(f"CREATING NODE ON {len(X_train)} EXAMPLES (depth: {depth})", end='\r')
        
        if depth > self.depth:
            self.depth = depth

        if self._pure_node(y_train):
            print("** Pure node, skipping **", end='\r')
        elif possible_attributes == []:
            print("** No more features, skipping **", end='\r')
        elif depth == self.tree_depth_limit:
            print("** At depth limit, skipping **", end='\r')
        
        if self._pure_node(y_train) or possible_attributes == [] or depth == self.tree_depth_limit or temp_depth == 0:
            self._make_majority_classifier(y_train, current_node)
        else: # prepare to choose a feature to partition on:
            if depth+2 <= tree_depth_limit and temp_depth is None and len(X_val) != 0: # ==> there is room left to try the research procedure (and we are not on a temporary tree)
                # The research procedure: we create three temporary trees of depth 2 on the 3 correspondingly highest features, to see if one outperforms the others.
                # We choose one and proceed with it based on accuracy measurement (on val set)
                information_measures, best_attribute_indices, best_attribute_thresholds = self._determine_split_criterion(X_train, y_train, possible_attributes, k_best=3)
                if information_measures[0] == 0 or information_measures == None: # ==> Max IG(X) or Max GR(X) = 0
                    self._make_majority_classifier(y_train, current_node)
                else:
                    # This part will build a small temporary decision tree out to depth+2 starting with the top three features as the test on the current node, then choose the one with the highest accuracy on the validations set:
                    max_accuracy = 0
                    best_attribute_index_after_validation_testing = None
                    best_attribute_threshold_after_validation_testing = None
                    for information_measure, best_attribute_index, best_attribute_threshold in zip(information_measures, best_attribute_indices, best_attribute_thresholds):
                        if self.schema[best_attribute_index].ftype == FeatureType.CONTINUOUS:
                            # Note: we do not update the possible attributes list here bc continuous tests may be made again on different thresholds.
                            # Continuous Partition procedure:
                            current_node.attribute_index = best_attribute_index
                            current_node.threshold = best_attribute_threshold

                            child_one, child_two = TreeNode(), TreeNode()
                            current_node.children = []
                            current_node.children.extend([child_one, child_two])

                            # Partitioning for less than or equal to the threshold
                            X_partition_leq, Y_partition_leq = self._partition_continuous(X_train, y_train, best_attribute_index, best_attribute_threshold, leq=True)
                            self._build_tree_research(X_partition_leq, Y_partition_leq, X_val, y_val, possible_attributes, child_one, depth+1, temp_depth=1)
                            # Partitioning for greater than the threshold
                            X_partition_g, Y_partition_g = self._partition_continuous(X_train, y_train, best_attribute_index, best_attribute_threshold, leq=False)
                            self._build_tree_research(X_partition_g, Y_partition_g, X_val, y_val,possible_attributes, child_two, depth+1, temp_depth=1)
                        else: 
                            # Nominal Partition procedure:
                            possible_attributes_updated = [i for i in possible_attributes if i != best_attribute_index] # (removed feature from possible_attributes)
                            current_node.attribute_index = best_attribute_index
                            current_node.threshold = None
                            current_node.children = []
                            current_node.nominal_values = []
                            for value in self.schema[best_attribute_index].values:
                                child = TreeNode()
                                current_node.nominal_values.append(value)
                                current_node.children.append(child)
                                X_partition, Y_partition = self._partition_nominal(X_train, y_train, best_attribute_index, value)
                                self._build_tree_research(X_partition, Y_partition, X_val, y_val,possible_attributes_updated, child, depth+1, temp_depth = 1)
                        # Now calculating accuracy on validation set:
                        y_hat = self.predict(X_val)
                        accuracy = util.accuracy(y_val, y_hat)
                        if accuracy >= max_accuracy:
                            max_accuracy = accuracy
                            best_attribute_index_after_validation_testing = best_attribute_index
                            best_attribute_threshold_after_validation_testing = best_attribute_threshold
                            
                    # Now that we know the best test, we will set the current_node to that test and call _build_tree_research again, with train and val sets partitioned on that test:
                    if self.schema[best_attribute_index_after_validation_testing].ftype == FeatureType.CONTINUOUS:
                        # Note: we do not update the possible attributes list here bc continuous tests may be made again on different thresholds.
                        # Continuous Partition procedure:
                        current_node.attribute_index = best_attribute_index_after_validation_testing
                        current_node.threshold = best_attribute_threshold_after_validation_testing

                        child_one, child_two = TreeNode(), TreeNode()
                        current_node.children = []
                        current_node.children.extend([child_one, child_two])
                        self.size += 2

                        # Partitioning for less than or equal to the threshold
                        X_train_partition_leq, y_train_partition_leq = self._partition_continuous(X_train, y_train, best_attribute_index_after_validation_testing, best_attribute_threshold_after_validation_testing, leq=True)
                        X_val_partition_leq, y_val_partition_leq = self._partition_continuous(X_val, y_val, best_attribute_index_after_validation_testing, best_attribute_threshold_after_validation_testing, leq=True)
                        self._build_tree_research(X_train_partition_leq, y_train_partition_leq,X_val_partition_leq, y_val_partition_leq, possible_attributes, child_one, depth+1)
                        # Partitioning for greater than the threshold
                        X_train_partition_g, y_train_partition_g = self._partition_continuous(X_train, y_train, best_attribute_index_after_validation_testing, best_attribute_threshold_after_validation_testing, leq=False)
                        X_val_partition_g, y_val_partition_g = self._partition_continuous(X_val, y_val, best_attribute_index_after_validation_testing, best_attribute_threshold_after_validation_testing, leq=False)
                        self._build_tree_research(X_train_partition_g, y_train_partition_g,X_val_partition_g, y_val_partition_g, possible_attributes, child_one, depth+1)
                    else: 
                        # Nominal Partition procedure:
                        possible_attributes_updated = [i for i in possible_attributes if i != best_attribute_index_after_validation_testing] # (removed feature from possible_attributes)
                        current_node.attribute_index = best_attribute_index_after_validation_testing
                        current_node.threshold = None
                        current_node.children = []
                        current_node.nominal_values = []
                        for value in self.schema[best_attribute_index_after_validation_testing].values:
                            self.size += 1
                            child = TreeNode()
                            current_node.nominal_values.append(value)
                            current_node.children.append(child)
                            X_train_partition, y_train_partition = self._partition_nominal(X_train, y_train, best_attribute_index_after_validation_testing, value)
                            X_val_partition, y_val_partition = self._partition_nominal(X_val, y_val, best_attribute_index_after_validation_testing, value)
                            self._build_tree_research(X_train_partition, y_train_partition, X_val_partition, y_val_partition, possible_attributes_updated, child, depth+1)
                ### partition val set too
            elif temp_depth is not None and temp_depth == 1:
                # this is the case where we are building out the intermediate layer of a research tree
                best_attribute_index, best_attribute_threshold = self._determine_split_criterion(X_train, y_train, possible_attributes)
                # create children and partition
                if self.schema[best_attribute_index].ftype == FeatureType.CONTINUOUS:
                    # Note: we do not update the possible attributes list here bc continuous tests may be made again on different thresholds.
                    # Continuous Partition procedure:
                    print("Assigning node continuous attribute " + self.schema[best_attribute_index].name + ", value " + str(best_attribute_threshold) + " at depth (" + str(depth) + ") ", end="\r")
                    current_node.attribute_index = best_attribute_index
                    current_node.threshold = best_attribute_threshold

                    child_one, child_two = TreeNode(), TreeNode()
                    current_node.children = []
                    current_node.children.extend([child_one, child_two])
                    
                    # Partitioning for less than or equal to the threshold
                    X_partition_leq, Y_partition_leq = self._partition_continuous(X_train, y_train, best_attribute_index, best_attribute_threshold, leq=True)
                    self._build_tree_research(X_partition_leq, Y_partition_leq, X_val, y_val,possible_attributes, child_one, depth+1, temp_depth=0)
                    # Partitioning for greater than the threshold
                    X_partition_g, Y_partition_g = self._partition_continuous(X_train, y_train, best_attribute_index, best_attribute_threshold, leq=False)
                    self._build_tree_research(X_partition_g, Y_partition_g, X_val, y_val, possible_attributes, child_two, depth+1, temp_depth=0)
                else: 
                    # Nominal Partition procedure:
                    possible_attributes_updated = [i for i in possible_attributes if i != best_attribute_index] # (removed feature from possible_attributes)
                    current_node.attribute_index = best_attribute_index
                    current_node.children = []
                    current_node.nominal_values = []
                    for value in self.schema[best_attribute_index].values:
                        child = TreeNode()
                        current_node.nominal_values.append(value)
                        current_node.children.append(child)
                        
                        X_partition, Y_partition = self._partition_nominal(X_train, y_train, best_attribute_index, value)
                        self._build_tree_research(X_partition, Y_partition, X_val, y_val,possible_attributes_updated, child, depth+1, temp_depth=0)
            else: 
                # then, there's only one possible depth level left in the tree, or we're out of validation examples to get accuracy data on so we don't run the research algorithm, we finish in the regular fashion:
                # thus, the validation set is now discarded
                best_attribute_index, best_attribute_threshold = self._determine_split_criterion(X_train, y_train, possible_attributes)
                # create children and partition
                if self.schema[best_attribute_index].ftype == FeatureType.CONTINUOUS:
                    # Note: we do not update the possible attributes list here bc continuous tests may be made again on different thresholds.
                    # Continuous Partition procedure:
                    print("Assigning node continuous attribute " + self.schema[best_attribute_index].name + ", value " + str(best_attribute_threshold) + " at depth (" + str(depth) + ") ", end="\r")
                    current_node.attribute_index = best_attribute_index
                    current_node.threshold = best_attribute_threshold

                    child_one, child_two = TreeNode(), TreeNode()
                    current_node.children = []
                    current_node.children.extend([child_one, child_two])
                    self.size += 2
                    
                    # Partitioning for less than or equal to the threshold
                    X_partition_leq, Y_partition_leq = self._partition_continuous(X_train, y_train, best_attribute_index, best_attribute_threshold, leq=True)
                    self._build_tree(X_partition_leq, Y_partition_leq, possible_attributes, child_one, depth+1)
                    # Partitioning for greater than the threshold
                    X_partition_g, Y_partition_g = self._partition_continuous(X_train, y_train, best_attribute_index, best_attribute_threshold, leq=False)
                    self._build_tree(X_partition_g, Y_partition_g, possible_attributes, child_two, depth+1)
                else: 
                    # Nominal Partition procedure:
                    possible_attributes_updated = [i for i in possible_attributes if i != best_attribute_index] # (removed feature from possible_attributes)
                    current_node.attribute_index = best_attribute_index
                    current_node.children = []
                    current_node.nominal_values = []
                    for value in self.schema[best_attribute_index].values:
                        child = TreeNode()
                        current_node.nominal_values.append(value)
                        current_node.children.append(child)
                        self.size += 1
                        
                        X_partition, Y_partition = self._partition_nominal(X_train, y_train, best_attribute_index, value)
                        self._build_tree(X_partition, Y_partition, possible_attributes_updated, child, depth+1,)

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
        if n_zeros == 0 or n_ones == 0: # then node is pure
            return True
        else:
            return False

    def _partition_nominal(self, X: np.ndarray, y: np.ndarray, feature_index, value) -> Tuple[np.ndarray, np.ndarray]:
        """
        A method for partitioning a subset of training data.

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
        A method for partitioning a subset of training data.

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

    def _determine_split_criterion(self, X: np.ndarray, y: np.ndarray, possible_attributes: np.ndarray, k_best: Optional[int]=None) -> Tuple[int, float]:
        """
        Args:
            X: The dataset. The shape is (n_examples, n_features).
            y: The labels. The shape is (n_examples,)
            possible_attributes: The list of the indices of the schema's features that are left to choose from at this node.
            k_best: How many of the top attribute indices to return.  If this is given, instead of None, this implies that it is being called from a research function  

        Returns: best_attribute_index, the schema index of the best possible attribute by the given metric, and best_threshold, a corresponding threshold value for that attribute if it is continuous (= None otherwise).  Could be arrays of top several if k_best != None (research extension).
        """
        # A boolean to signify whether or not this is for the research extension or not
        
        max_information_measure = 0
        best_attribute_index = None
        best_threshold = None

        # (for research extension) setting this function up to return top few attributes in an array instead of the single best:  
        research = False
        if k_best is not None:
            research = True
            max_information_measure = []
            best_attribute_index = []
            best_threshold = []

        H_y = util.entropy(y)

        for index in possible_attributes:
            feature = self._schema[index]
            if feature.ftype == FeatureType.CONTINUOUS:
                # helper function that finds all possible thresholds
                dividers = self._find_thresholds_2(X[:, index], y)
                print(f"Attribute {feature.name} has {len(dividers)} potential thresholds", end='\r')
                
                best_information_measure_per_attribute = 0

                for div in dividers:
                    if self.use_information_gain:
                        current_IG = H_y - util.conditional_entropy(X, y, index, div)
                        if not research:
                            if current_IG > max_information_measure:
                                max_information_measure = current_IG
                                best_attribute_index = index
                                best_threshold = div
                        else:
                            max_information_measure.append(current_IG)
                            best_attribute_index.append(index)
                            best_threshold.append(div)
                        if current_IG > best_information_measure_per_attribute:
                            best_information_measure_per_attribute = current_IG
                    else:
                        current_GR = (H_y - util.conditional_entropy(X, y, index, div)) / util.attribute_entropy(X, index, div)
                        if not research:
                            if current_GR > max_information_measure:
                                max_information_measure = current_GR
                                best_attribute_index = index
                                best_threshold = div
                        else:
                            max_information_measure.append(current_GR)
                            best_attribute_index.append(index)
                            best_threshold.append(div)
                        if current_GR > best_information_measure_per_attribute:
                            best_information_measure_per_attribute = current_GR
                
                if self.use_information_gain:
                    print(f"Checked continuous attribute {feature.name} ({index})...    IG = {best_information_measure_per_attribute}", end="\r")
                else:
                    print(f"Checked continuous attribute {feature.name} ({index})...    GR = {best_information_measure_per_attribute}", end="\r")

            else:
                if self.use_information_gain:
                    current_IG = H_y - util.conditional_entropy(X, y, index, None)
                    print(f"Checking nominal attribute {feature.name} ({index})...    IG = {current_IG}", end="\r")
                    if not research:
                        if current_IG > max_information_measure:
                            max_information_measure = current_IG
                            best_attribute_index = index
                    else:
                        max_information_measure.append(current_IG)
                        best_attribute_index.append(index)
                        best_threshold.append(None)
                else:
                    current_GR = (H_y - util.conditional_entropy(X, y, index, None)) / util.attribute_entropy(X, index, None)
                    print(f"Checking nominal attribute {feature.name} ({index})...    GR = {current_GR}", end="\r")
                    if not research:
                        if current_GR > max_information_measure:
                            max_information_measure = current_GR
                            best_attribute_index = index
                    else:
                        max_information_measure.append(current_GR)
                        best_attribute_index.append(index)
                        best_threshold.append(None)
        if research:
            # how to pick the top k_best out of the constructed arrays:
            zipped = zip(max_information_measure, best_attribute_index, best_threshold)
            zipped = list(zipped)
            sorted_zipped = sorted(zipped, key = lambda x: x[0], reverse=True)
            last_index = min(k_best, len(possible_attributes)) # for picking top k_best unless there are less possible attributes than k_best
            max_information_measure, best_attribute_index, best_threshold = list(zip(*sorted_zipped[0:last_index]))
            max_information_measure = list(max_information_measure)
            best_attribute_index = list(best_attribute_index)
            best_threshold = list(best_threshold)
            return max_information_measure, best_attribute_index, best_threshold
        else:
            return best_attribute_index, best_threshold

    def _find_thresholds(self, values: np.ndarray) -> np.ndarray:
        """
        Find a set of candidate thresholds that could be used to partition a continuous variable.
        In reality, this function just returns the set of unique values of a continuous variable in a set of examples.
        Even though the set of 'viable' thresholds is much smaller (since only thresholds between values with
        differing class labels should be considered), *finding thresholds this way is actually far more efficient*.
        This is because the 'proper' method requires sorting the array of seen values, which can be 10000s of elements long.
        
        Args:
            values: An array of seen values for a given continuous attribute.

        Returns: An array of thresholds to check for partitioning.
        """
        return np.unique(values)

    def _find_thresholds_2(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Find a set of candidate thresholds that could be used to partition a continuous variable.
        This function implements the 'proper' method for finding partitioning thresholds:
        it only returns thresholds between adjacent values with different class labels.

        Args:
            values: An array of seen values for a given continuous attribute.
            labels: An array of class labels associated with those values.

        Returns: An array of thresholds to check for partitioning.
        """ 
        values_sort = [[v, l] for v, l in zip(values, labels)]
        values_sort = sorted(values_sort, key=lambda x: x[0])

        dividers = [values_sort[0][0] - 0.01]
        for i in range(1, len(labels)):
            if values_sort[i][1] != values_sort[i-1][1]:
                dividers.append((values_sort[i][0] + values_sort[i-1][0]) / 2)
        dividers.append(values_sort[-1][0] + 0.01)
        
        return np.unique(dividers)


def evaluate_and_print_metrics(dtree: DecisionTree, X: np.ndarray, y: np.ndarray):
    """
    Print information about the performance of a decision tree on testing data.

    Args:
        dtree: The DecisionTree instance to evaluate.
        X: An array of examples to use for evaluation.
        y: An array of class labels associated with the examples in X.
    """

    y_hat = dtree.predict(X)
    acc = util.accuracy(y, y_hat)
    print("\n***********\n* RESULTS *\n***********")
    print(f'Accuracy:{acc:.2f}')
    print('Size:', dtree.size)
    print('Maximum Depth:', dtree.depth)
    print('First Feature:', dtree.schema[dtree.root.attribute_index].name, '\n')


def dtree(data_path: str, tree_depth_limit: int, use_cross_validation: bool = True, information_gain: bool = True, research: bool = False):
    """
    Create and train decision trees on data in a given folder.

    Args: 
        data_path: The path to the data.
        tree_depth_limit: Depth limit of the decision tree
        use_cross_validation: If True, use cross validation. Otherwise, run on the full dataset.
        information_gain: If true, use information gain as the split criterion. Otherwise, use gain ratio.
        research: If true, call the research functions (for part (d)) instead of the original, basic functions.
    """

    # last entry in the data_path is the file base (name of the dataset)
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
    root_dir = os.sep.join(path[:-1])
    schema, X, y = parse_c45(file_base, root_dir)

    if use_cross_validation:
        datasets = util.cv_split(X, y, folds=5, stratified=True)
    else:
        datasets = ((X, y, X, y),)

    if not research:
        for X_train, y_train, X_test, y_test in datasets:
            decision_tree = DecisionTree(schema,
                    tree_depth_limit=tree_depth_limit,
                    use_information_gain=information_gain)
            decision_tree.fit(X_train, y_train)
            evaluate_and_print_metrics(decision_tree, X_test, y_test)
    else:
        for X_train, y_train, X_test, y_test in datasets:
            # Creating a validation set from the training set by using cv_split
            X_train, y_train, X_val, y_val = util.cv_split(X_train, y_train, folds=5, stratified=True)[0]  
            decision_tree = DecisionTree(schema,
                tree_depth_limit=tree_depth_limit,
                use_information_gain=information_gain)
            decision_tree.fit(X_train, y_train, X_val, y_val, research=True)
            evaluate_and_print_metrics(decision_tree, X_test, y_test)


if __name__ == '__main__':
    """
    Main method.  
    
    Parses run arguments, then calls dtree with formatted arguments.
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
    parser.add_argument('--research', dest='research', action='store_true',
                        help='Enables the fit() and tree algo for the research question instead of running the normal fit/tree algo.')
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
    research = args.research

    dtree(data_path, tree_depth_limit, use_cross_validation, use_information_gain, research)
