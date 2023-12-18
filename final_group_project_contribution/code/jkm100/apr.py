"""
Created on 11/30/22 by John Mays

This is my implementation of a "standard" Axis-Parallel Rectangles (APR) Algorithm for the problem of multiple-instance classification.  This is a version of axis-parallel rectangles described in Dietterich et al.'s 1997 paper "Solving the multiple-instance problem with axis-parallel rectangles."
"""
import sys
sys.path.append(".")

from group.classifier import MILClassifier
from group.util import accuracy, unbag_MIL_data
from typing import List

import numpy as np
import random

random.seed(0)

class APRClassifier(MILClassifier):
    def __init__(self, allpos = False):
        self.bounds = None
        self.allpos = allpos

    def fit(self, X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray):
        self.bounds = np.zeros((np.shape(X)[1], 2)) # creating bounds matrix
        X, y_instances = unbag_MIL_data(X, bag_indices, y) # unbagging bags

        # Finding the "all-positive APR":
        for feature_index in range(np.shape(X)[1]):
            max_value = -np.inf # (these are the max and min feature value belonging to a positive instance)
            min_value = np.inf 
            for x, label in zip(X, y_instances):
                if label == 1:
                    current_value = x[feature_index]
                    if current_value > max_value:
                        max_value = current_value
                    if current_value < min_value:
                        min_value = current_value
            self.bounds[feature_index, 0] = min_value
            self.bounds[feature_index, 1] = max_value
        # Calling procedure to narrow bounds:
        if not self.allpos:
            # self.elim_count(X, y_instances)
            self.greedy_feature_selection(X, y_instances)

    def greedy_feature_selection(self, X, y_instances):
        # saving the weights in a variable and then setting the weights to the entire feature space
        entire_feature_space = np.vstack((-np.inf*np.ones(np.shape(X)[1]), np.inf*np.ones(np.shape(X)[1]))).T
        potential_bounds = np.copy(self.bounds)
        self.bounds = entire_feature_space
        added_back = np.zeros(np.shape(self.bounds), dtype=bool)

        y_instances_hat = self.predict_instances(X)
        num_false_positives = np.sum((y_instances==0)*(y_instances_hat==1))
        prev_num_false_positives = np.inf
        while(num_false_positives > 0 and num_false_positives < prev_num_false_positives): # (while any true negatives are classified as positives)
            prev_num_false_positives = num_false_positives
            print(f'num examples in pos region: {np.sum(y_instances_hat)}')
            print(f'in GFS, num negs misclassified: {num_false_positives}')
            print(f'num boundaries untouched: {np.sum(np.abs(self.bounds)==np.inf)}')
            ### print(f'num elements in bounds inf: {np.sum(np.abs(self.bounds)==np.inf)}')
            # iterate through bounds:
            num_false_positives_eliminated_min = np.zeros(np.shape(potential_bounds[:,0]))
            num_false_positives_eliminated_max = np.zeros(np.shape(potential_bounds[:,1]))
            for i, (potential_min_bound, potential_max_bound) in enumerate(zip(potential_bounds[:,0],potential_bounds[:,1])):
                # min:
                if added_back[i,0] == False:
                    self.bounds[i,0] = potential_min_bound
                    y_instances_hat_temp = self.predict_instances(X)
                    num_false_positives_eliminated_min[i] = np.sum((y_instances==0)*(y_instances_hat==1)) - np.sum((y_instances==0)*(y_instances_hat_temp==1)) 
                    self.bounds[i,0] = -np.inf
                # max:
                if added_back[i,1] == False:
                    self.bounds[i,1] = potential_max_bound
                    y_instances_hat_temp = self.predict_instances(X)
                    num_false_positives_eliminated_max[i] = np.sum((y_instances==0)*(y_instances_hat==1)) - np.sum((y_instances==0)*(y_instances_hat_temp==1))
                    self.bounds[i,1] = np.inf
            # choose best & actually add it & set add back to true
            best_min_bound_index = np.argmax(num_false_positives_eliminated_min)
            most_false_postivies_eliminated_by_min = num_false_positives_eliminated_min[best_min_bound_index]
            best_max_bound_index = np.argmax(num_false_positives_eliminated_max)
            most_false_postivies_eliminated_by_max = num_false_positives_eliminated_max[best_max_bound_index]
            if most_false_postivies_eliminated_by_min > most_false_postivies_eliminated_by_max:
                self.bounds[best_min_bound_index,0] = potential_bounds[best_min_bound_index,0]
                added_back[best_min_bound_index,0] = True
                ### print(f'feature idx being changed: {best_min_bound_index}')
            else:
                self.bounds[best_max_bound_index,1] = potential_bounds[best_max_bound_index,1]
                added_back[best_max_bound_index,1] = True
                ### print(f'feature idx being changed: {best_max_bound_index}')
            y_instances_hat = self.predict_instances(X)
            num_false_positives = np.sum((y_instances==0)*(y_instances_hat==1))
 
    def predict_instances(self, X: np.ndarray) -> np.ndarray:
        # in other words, instance will only be classified as positive if all of its values are within or on bounds
        y_hat = np.all(np.greater_equal(X-self.bounds[:,0], 0), axis=1) * np.all(np.less_equal(X-self.bounds[:,1], 0), axis=1)
        return y_hat

    def predict(self, X: np.ndarray, bag_indices: List[np.ndarray]) -> np.ndarray:
        """
        Runs APR prediction on bags instead of instances.
        Uses the SMI assumption: A bag is positive iff. it has >= 1 positive instances.
        (A more appropriate name for this function would be predict_bags (as opposed to my predict_instances), but I am complying with my group's abstract class.)
        
        Args:
        X: examples matrix
        bag_indices: list of lists that represent bags.  Elements of sublists are indices of X.

        Returns:
            y_hat: predictions vector (0s and 1s) as an np.ndarray with same size of y.
        """
        # making instance predictions first:
        y_instances_hat = self.predict_instances(X)

        # initializing predictions vector:
        y_hat = np.zeros(len(bag_indices))

        # using instance predictions to make bag predictions:
        for bag_index, indices in enumerate(bag_indices):
            for index in indices:
                if y_instances_hat[index] == 1:
                    y_hat[bag_index] = 1
        return y_hat