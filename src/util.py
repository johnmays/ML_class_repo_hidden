from multiprocessing.resource_tracker import getfd
import random
from turtle import shape
import warnings
from typing import Tuple, Iterable

import numpy as np

"""
This is where you will implement helper functions and utility code which you will reuse from project to project.
Feel free to edit the parameters if necessary or if it makes it more convenient.
Make sure you read the instruction clearly to know whether you have to implement a function for a specific assignment.
"""


def count_label_occurrences(y: np.ndarray) -> Tuple[int, int]:
    """
    This method simply takes an array of labels and counts the number of positive and negative labels.

    Args:
        y: Array of binary labels.

    Returns: A tuple containing the number of negative occurrences, and number of positive occurences, respectively.
    """

    n_ones = np.linalg.norm(y, ord=1) 
    return len(y) - n_ones, n_ones


def majority_class(y: np.ndarray) -> int:
    n_zeros, n_ones = count_label_occurrences(y)
    return 0 if n_zeros > n_ones else 1


def entropy(y: np.ndarray) -> float:
    """
    Returns the Shannon entropy of a node with a given set of remaining BINARY class labels.

    Args:
        y: A list of class labels for the training examples associated with a node.

    Returns: The Shannon entropy of the node.
    """
    if len(y) == 0:
        return 0
    n_zeros, n_ones = count_label_occurrences(y)
    p_zero, p_one = n_zeros / len(y), n_ones / len(y)
    if p_zero == 0 or p_one == 0:
        return 0
    return (-p_zero * np.log2(p_zero)) + (-p_one * np.log2(p_one))


def entropy_nb(y: np.ndarray) -> float:
    """
    Returns the Shannon entropy of a variable with the given set of NON-BINARY values.

    Args:
        y: A list of examples of values for the variable.

    Returns: The Shannon entropy of the variable.
    """
    values = {}
    for val in y:
        if val in values:
            values[val] += 1
        else:
            values[val] = 1
    
    H = 0
    for val in values:
        p = values[val] / len(y)
        H += -p * np.log2(p)
    return H


def conditional_entropy(X: np.ndarray, y: np.ndarray, index: int, threshold: float) -> float:
    """
    Returns the information gain for partitioning on an attribute, 
    given a set of training examples and class labels associated with a node.

    Args:
        X: The values of attributes for the training examples at a node
        Y: The class labels associated with those examples
        index: The index of the attribute being partitioned on
        threshold: The value of the attribute to split by, if the attribute is continuous.
            Should be None if the index is nominal.
    
    Returns: The information gain by partitioning the examples on the given attribute test.
    """

    H_y_given_x = 0 # the entropy of the node after partitioning 

    values_for_attribute = X[:, index]
    if threshold is None:
        # Create a dictionary that maps each nominal value to a list of class labels
        # for examples with that value
        ex_nominal = {}
        for i in range(len(y)):
            if values_for_attribute[i] in ex_nominal:
                ex_nominal[values_for_attribute[i]].append(y[i])
            else:
                ex_nominal[values_for_attribute[i]] = []
        for v, labels in ex_nominal.items():
            H_y_given_x += (len(labels) / len(y)) * entropy(labels)
            
    else:
        ex_less_than_equal = []
        ex_greater_than = []
        for i in range(len(y)):
            if values_for_attribute[i] <= threshold:
                ex_less_than_equal.append(y[i])
            else:
                ex_greater_than.append(y[i])
        H_y_given_x = (len(ex_less_than_equal) * entropy(ex_less_than_equal) / len(y)) + (len(ex_greater_than) * entropy(ex_greater_than) / len(y))
        
    return H_y_given_x


def gain_ratio(X: np.ndarray, y: np.ndarray, index: int, threshold: float) -> float:
    """
    Returns the gain ratio for partitioning a node with given examples
    on a given attribute test.
    
    Args:
        X: The values of attributes for the training examples at a node
        Y: The class labels associated with those examples
        index: The index of the attribute being partitioned on
        threshold: The value of the attribute to split by, if the attribute is continuous.
            Should be None if the index is nominal.
    
    Returns: The gain ratio for partitioning the examples on the given attribute test.
    """
    ig = entropy(y) - conditional_entropy(X, y, index, threshold)

    if threshold is None:
        branches = X[:, index]
    else:
        branches = [(x <= threshold) for x in X[:, index]]
    H_x = entropy_nb(branches)

    return ig / H_x


def cv_split(
        X: np.ndarray, y: np.ndarray, folds: int, stratified: bool = False
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
    """
    Conducts a cross-validation split on the given data.

    Args:
        X: Data of shape (n_examples, n_features)
        y: Labels of shape (n_examples,)
        folds: Number of CV folds
        stratified:

    Returns: A tuple containing the training data, training labels, testing data, and testing labels, respectively
    for each fold.

    For example, 5 fold cross validation would return the following:
    (
        (X_train_1, y_train_1, X_test_1, y_test_1),
        (X_train_2, y_train_2, X_test_2, y_test_2),
        (X_train_3, y_train_3, X_test_3, y_test_3),
        (X_train_4, y_train_4, X_test_4, y_test_4),
        (X_train_5, y_train_5, X_test_5, y_test_5)
    )

    """

    # Set the RNG seed to 12345 to ensure repeatability
    np.random.seed(12345)
    random.seed(12345)

    # HINT!
    if stratified:
        n_zeros, n_ones = count_label_occurrences(y)

    # Copy X and y
    X = X.copy()
    y = y.copy()
    tup = ()
    # Find fold size
    foldsize = int(y.size / folds)

    # Split the ndarray randomly into n fold stratified
    if stratified:
        X_one = X[np.where(y == 1)].copy()
        X_zero = X[np.where(y == 0)].copy()
        y_one = y[np.where(y == 1)].copy()
        y_zero = y[np.where(y == 0)].copy()
        ones = int(len(X_one)/folds)
        zeros = int(len(X_zero)/folds)
        remainder_one = len(X_one)%folds
        remainder_zero = len(X_zero)%folds
        for f in range(0, folds-1):
            if f < remainder_one:
                result_x_one, result_y_one, remain_x_one, remain_y_one = getData(X_one, y_one, ones+1)
            else:
                result_x_one, result_y_one, remain_x_one, remain_y_one = getData(X_one, y_one, ones)
            if f < remainder_zero:
                result_x_zero, result_y_zero, remain_x_zero, remain_y_zero = getData(X_zero, y_zero, zeros+1)
            else:
                result_x_zero, result_y_zero, remain_x_zero, remain_y_zero = getData(X_zero, y_zero, zeros)
            
            if len(result_x_one) == 0:
                result_x = result_x_zero
            elif len(result_x_zero) == 0:
                result_x = result_x_one
            else:
                result_x = np.append(result_x_one, result_x_zero, axis=0)

            if len(result_y_one) == 0:
                result_y = result_y_zero
            elif len(result_y_zero) == 0:
                result_y = result_y_one
            else:
                result_y = np.append(result_y_one, result_y_zero)


            tup += ((result_x, result_y),)
            X_one = remain_x_one
            X_zero = remain_x_zero
            y_one = remain_y_one
            y_zero = remain_y_zero
        if len(X_one) == 0:
            result_x = X_zero
        elif len(X_zero) == 0:
            result_x = X_one
        else:
            result_x = np.append(X_one, X_zero, axis=0)

        if len(y_one) == 0:
            result_y = y_zero
        elif len(y_zero) == 0:
            result_y = y_one
        else:
            result_y = np.append(y_one, y_zero)
        tup += ((result_x, result_y),)
        print(tup)
                


    # Split the ndarray ramdomly into n fold non-stratified        
    else:
        # Compute the remainder if examples cannot evenly distribute to each fold
        remainder = len(y)%folds
        # Compute the first folds-1 fold
        for f in range(0, folds-1):
            # If there are remainders then add one extra example to first remainder fold
            if f < remainder:
                result_x, result_y, remain_x, remain_y = getData(X, y, foldsize+1)
                tup += ((result_x, result_y),)
                X = remain_x
                y = remain_y
                
            else:
                result_x, result_y, remain_x, remain_y = getData(X, y, foldsize)
                tup += ((result_x, result_y),)
                X = remain_x
                y = remain_y
                
        # Append the rest of ndarray as the last fold
        tup += ((X,y),)
        print(tup)
    # Combine the folds into n sets
    result = ()
    for i in range(0, len(tup)):
        test_x = tup[i][0]
        test_y = tup[i][1]

        train_x = [tup[i][0][0]]
        temp_x = len(tup[i][0])
        train_y = [tup[i][1][0]]
        temp_y = len(tup[i][1])
        for ind, val in enumerate(tup):
            if ind == i:
                continue
            train_x = np.append(train_x, val[0].copy(), axis=0)  
            train_y = np.append(train_y, val[1].copy())
        train_x = np.delete(train_x, 0, axis=0)
        train_y = np.delete(train_y, 0)
        result += ((train_x, train_y, test_x, test_y),)
    
    return result


        

    warnings.warn('cv_split is not yet implemented. Simply returning the entire dataset as a single fold...')

    return (X, y, X, y),


def getData(X: np.ndarray, y:np.ndarray, num: int):
    """
    Select random element from given array, remove element when extract

    Args: 
        array: array of raw data
        num: number of element need to be extract from array

    Returns:
        result: array of extracted data
        array: the origional array after extract data
    """
    if num == 0:
        return X, y, [], []
    # Set random seed
    np.random.seed(12345)
    random.seed(12345)

    # Create framework of result
    index = random.randint(0, len(X)-1)

    result_x = np.array([X[index].copy()])
    result_y = np.array([y[index].copy()])
    X = np.delete(X, index, axis=0)
    y = np.delete(y, index)
    
    # Extract element from origional array
    for i in range(0, num-1):
        index = random.randint(0, len(X)-1)
        result_y = np.append(result_y, y[index].copy())
        y = np.delete(y, index)
        
        result_x = np.append(result_x, [X[index].copy()], axis=0)
        X = np.delete(X, index, axis=0)

    return result_x, result_y, X, y



def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Another example of a helper method. Implement the rest yourself!

    Args:
        y: True labels.
        y_hat: Predicted labels.

    Returns: Accuracy
    """

    if len(y) != len(y_hat):
        raise ValueError('y and y_hat must be the same shape/size!')

    n = len(y)

    return (y == y_hat).sum() / n


def precision(y: np.ndarray, y_hat: np.ndarray) -> float:
    raise NotImplementedError()


def recall(y: np.ndarray, y_hat: np.ndarray) -> float:
    raise NotImplementedError()


def roc_curve_pairs(y: np.ndarray, p_y_hat: np.ndarray) -> Iterable[Tuple[float, float]]:
    raise NotImplementedError()


def auc(y: np.ndarray, p_y_hat: np.ndarray) -> float:
    roc_pairs = roc_curve_pairs(y, p_y_hat)
    raise NotImplementedError()