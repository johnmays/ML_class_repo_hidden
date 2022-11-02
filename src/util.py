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
    """
    Returns the majority class out of a list of class labels.

    Args:
        y: A list of class labels containing only 1s and 0s.
    
    Returns: The majority class in the list.
    """

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
    Returns the conditional entropy H(Y|X) for partitioning on an attribute, 
    given a set of training examples and class labels associated with a node.

    Args:
        X: The values of attributes for the training examples at a node
        Y: The class labels associated with those examples
        index: The index of the attribute being partitioned on
        threshold: The value of the attribute to split by, if the attribute is continuous.
            Should be None if the index is nominal.
    
    Returns: The conditional entropy by partitioning the examples on the given attribute test.
    """
    H_y_given_x = 0 # the entropy of the node after partitioning 

    values_for_attribute = np.array(X[:, index])
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
        # This code has been VERY heavily optimized
        if True:
            lte = values_for_attribute <= threshold
            gt = lte == False

            total_lte = np.sum(lte)
            total_gt = len(y) - total_lte

            ones_lte = np.sum(lte * (y == 1))
            ones_gt = np.sum(gt * (y == 1))

            if total_lte != 0:
                p_lte = ones_lte/total_lte
                if p_lte > 0 and p_lte < 1:
                    H_y_given_x += ((total_lte/len(y)) * (-p_lte*np.log2(p_lte) + -(1-p_lte)*np.log2(1-p_lte))) 
            if total_gt != 0:
                p_gt = ones_gt/total_gt
                if p_gt > 0 and p_gt < 1:
                    H_y_given_x += ((total_gt/len(y)) * (-p_gt*np.log2(p_gt) + -(1-p_gt)*np.log2(1-p_gt)))
        
    return H_y_given_x


def attribute_entropy(X: np.ndarray, index: int, threshold: float) -> float:
    """
    Returns the entropy H(i) of a given attribute i over a set of examples X.

    Args:
        X: A set of examples to check for values of an attribute.
        index: The index of the attribute to get the entropy of.

    Returns: The entropy of the attribute within the set of examples.
    """
    if threshold is None:
        branches = X[:, index]
    else:
        branches = [(x <= threshold) for x in X[:, index]]
    return entropy_nb(branches)


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
        #Split the examples into fold stratify
        #Split the examples into ones and zeros
        X_one = X[np.where(y == 1)].copy()
        X_zero = X[np.where(y == 0)].copy()
        y_one = y[np.where(y == 1)].copy()
        y_zero = y[np.where(y == 0)].copy()
        #Calculate number of ones and zeros in each fold 
        #Calculate number of remainders of ones and zeros
        ones = int(len(X_one)/folds)
        zeros = int(len(X_zero)/folds)
        remainder_one = len(X_one)%folds
        remainder_zero = len(X_zero)%folds
        #Evenly distribute extra examples to first X sets where X is the number of remainder
        for f in range(0, folds-1):
            if f < remainder_one:
                result_x_one, result_y_one, remain_x_one, remain_y_one = getData(X_one, y_one, ones+1)
            else:
                result_x_one, result_y_one, remain_x_one, remain_y_one = getData(X_one, y_one, ones)
            if f < remainder_zero:
                result_x_zero, result_y_zero, remain_x_zero, remain_y_zero = getData(X_zero, y_zero, zeros+1)
            else:
                result_x_zero, result_y_zero, remain_x_zero, remain_y_zero = getData(X_zero, y_zero, zeros)
            #If we run out of one label we just form the fold with the other label
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

            #Append the result to tup
            tup += ((result_x, result_y),)
            X_one = remain_x_one
            X_zero = remain_x_zero
            y_one = remain_y_one
            y_zero = remain_y_zero
        #Append the rest of example to the last fold
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
    Returns the accuracy between a true results array and a predicted array.

    Args:
        y: True labels.
        y_hat: Predicted labels.

    Returns: Accuracy = (TP+TN)/(TP+TN+FP+FN)
    """

    n = len(y)

    if n != len(y_hat):
        raise ValueError('y and y_hat must be the same shape/size!')

    return (y == y_hat).sum() / n


def precision(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Returns the precision between a true results array and a predicted array.

    Args:
        y: True labels.
        y_hat: Predicted labels.

    Returns: Precision = TP/(TP+FP)
    """

    n = len(y)
    
    if n != len(y_hat):
        raise ValueError('y and y_hat must be the same shape/size!')

    return ((y == y_hat)*(y==1)).sum() / (((y == y_hat)*(y==1)).sum() + ((y != y_hat)*(y_hat==1)).sum())


def recall(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Returns the recall between a true results array and a predicted array.

    Args:
        y: True labels.
        y_hat: Predicted labels.

    Returns: Recall = TPR = TP/(TP+FN)
    """

    n = len(y)
    
    if n != len(y_hat):
        raise ValueError('y and y_hat must be the same shape/size!')

    return ((y == y_hat)*(y==1)).sum() / (((y == y_hat)*(y==1)).sum() + ((y != y_hat)*(y_hat==0)).sum())

def false_positive_rate(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Returns the FPR between a true results array and a predicted array.

    Args:
        y: True labels.
        y_hat: Predicted labels.

    Returns: FPR = FP/(FP+TN)
    """

    n = len(y)
    
    if n != len(y_hat):
        raise ValueError('y and y_hat must be the same shape/size!')

    return ((y != y_hat)*(y==1)).sum() / (((y == y_hat)*(y==0)).sum() + ((y != y_hat)*(y_hat==1)).sum())

def roc_curve_pairs(y: np.ndarray, p_y_hat: np.ndarray) -> Iterable[Tuple[float, float]]:
    """
    Returns  DO LATER
    """
    assert np.shape(y) == np.shape(p_y_hat), 'Arguments must be the same size'
    sorted_pairs = sorted(zip(p_y_hat, y)) # zip and sort
    p_y_hat, y = zip(*sorted_pairs) # unzip
    pairs = []
    y_hat_confidence = np.ones(len(y)+1) # everything above the confidence threshold is 1. starts all ones
    for i in range(len(y)):
        if i != 0:
            y_hat_confidence[i-1] = 0 
        pairs.append((recall(y, y_hat_confidence), false_positive_rate(y, y_hat_confidence)))
    return pairs

def auc(y: np.ndarray, p_y_hat: np.ndarray) -> float:
    roc_pairs = roc_curve_pairs(y, p_y_hat)
    roc_pairs.sort(key = lambda x: x[0])

    area = 0
    last_pair = roc_pairs[0]
    for i in range(1, len(roc_pairs)):
        next_pair = roc_pairs[i]
        area += ((last_pair[1] + next_pair[1]) / 2) * (next_pair[0] - last_pair[0])
        last_pair = next_pair
    
    return area