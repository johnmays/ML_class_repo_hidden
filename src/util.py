import random
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
    This is a simple example of a helpful helper method you may decide to implement. Simply takes an array of labels and
    counts the number of positive and negative labels.

    HINT: Maybe a method like this is useful for calculating more complicated things like entropy!

    Args:
        y: Array of binary labels.

    Returns: A tuple containing the number of negative occurrences, and number of positive occurences, respectively.

    """
    n_ones = (y == 1).sum()  # How does this work? What does (y == 1) return?
    n_zeros = y.size - n_ones
    return n_zeros, n_ones


def entropy(p_x: np.ndarray):
    """
    Returns the Shannon entropy of a variable with the given probability mass values.

    Args:
        p_x: An array containing a set of probability mass values for a variable.

    Returns: The Shannon entropy of the variable.
    """
    return np.sum([-p * np.log2(p) for p in p_x])


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
    x_tup = ()
    y_tup = ()
    # Find fold size
    foldsize = int(y.size / folds)

    # Split the ndarray randomly into n fold stratified
    if stratified:
        X_one = X[np.where(X[-1] == 1)]
        X_zero = X[np.where(X[-1] == 0)]
        y_one = y[np.where(y[0] == 1)]
        y_zero = y[np.where(y[0] == 0)]

    # Split the ndarray ramdomly into n fold non-stratified        
    else:
        # Compute the remainder if examples cannot evenly distribute to each fold
        remainder = len(y)%folds
        # Compute the first folds-1 fold
        for f in range(0, folds-1):
            # If there are remainders then add one extra example to first remainder fold
            if f < remainder:
                result_x, remain_x = getData(X, foldsize+1)
                x_tup += (result_x,)
                X = remain_x

                result_y, remain_y = getData(y, foldsize+1)
                y_tup += (result_y,)
                y = remain_y
            else:
                result_x, remain_x = getData(X, foldsize)
                x_tup += (result_x,)
                X = remain_x

                result_y, remain_y = getData(y, foldsize)
                y_tup += (result_y,)
                y = remain_y
        # Append the rest of ndarray as the last fold
        x_tup += (X,)
        y_tup += (y,)

    # Combine the folds into n sets
    result = ()
    for a in range(0, folds):
        result += ((x_tup[0:a]+x_tup[a+1:], y_tup[0:a]+y_tup[a+1:], x_tup[a], y_tup[a]),)
    return result


        

    warnings.warn('cv_split is not yet implemented. Simply returning the entire dataset as a single fold...')

    return (X, y, X, y),


def getData(array: np.ndarray, num: int):
    """
    Select random element from given array, remove element when extract

    Args: 
        array: array of raw data
        num: number of element need to be extract from array

    Returns:
        result: array of extracted data
        array: the origional array after extract data
    """

    # Set random seed
    np.random.seed(12345)
    random.seed(12345)

    # Check if extracting y_lable
    isY = len(array.shape)

    # Create framework of result
    index = random.randint(0, len(array)-1)
    if isY == 1:
        result = [array[index].copy()]
        array = np.delete(array, index)
    else:
        result = [array[index].copy()]
        array = np.delete(array, index, axis=0)
    
    # Extract element from origional array
    for i in range(0, num-1):
        index = random.randint(0, len(array)-1)
        if isY == 1:
            result = np.append(result, array[index].copy())
            array = np.delete(array, index)
        else:
            result = np.append(result, [array[index].copy()], axis=0)
            array = np.delete(array, index, axis=0)

    return result, array



def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Another example of a helper method. Implement the rest yourself!

    Args:
        y: True labels.
        y_hat: Predicted labels.

    Returns: Accuracy
    """

    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')

    n = y.size

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