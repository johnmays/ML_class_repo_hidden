"""
Created on 12/09/22 by John Mays

"""
import sys
sys.path.append(".")

from group.classifier import MILClassifier
from group.util import accuracy, unbag_MIL_data, cv_split
from typing import List

from apr import APRClassifier

import numpy as np
import random

random.seed(0)

class BalancedAPRClassifier(APRClassifier):
    def __init__(self, lamb = None):
        self.bounds = None
        self.lamb = lamb
        self.lamb_was_set = True
        if self.lamb is None:
            self.lamb_was_set = False

    def fit(self, X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray):
        self.bounds = np.zeros((np.shape(X)[1], 2)) # creating bounds matrix
        X, y_instances = unbag_MIL_data(X, bag_indices, y) # unbagging bags
        if self.lamb_was_set == False:
            self.lamb = self.lamb_estimation(X, bag_indices, y, labeled_instances = False)
            print(f'A exchange value of {self.lamb} was chosen by lambda estimation...')

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
        self.balanced_elim_count(X, y_instances)
        # self.greedy_feature_selection(X, y_instances)

    def balanced_elim_count(self, X, y_instances):
        """
        Performs elim_count procedure with misclassification cost augmentation.
        """
        y_instances_hat = self.predict_instances(X)
        prev_y_instances_hat = 2*np.ones(np.shape(y_instances_hat)) # dummy vector != to any feasible predictions vector
        while(np.any((y_instances==0)*(y_instances_hat==1)) and not np.all(np.equal(y_instances_hat, prev_y_instances_hat))): # (while any true negatives are classified as positives) or while classification results are changing
            prev_y_instances_hat = y_instances_hat
            ### print(np.sum((y_instances==0)*(y_instances_hat==1)))
            # min bounds:
            misclassification_costs_min = np.zeros(np.size(self.bounds[:,0]))
            temp_bounds_min = np.zeros(np.size(self.bounds[:,0]))
            misclassification_costs_max = np.zeros(np.size(self.bounds[:,1]))
            temp_bounds_max = np.zeros(np.size(self.bounds[:,1]))

            for i, (min_bound, max_bound) in enumerate(zip(self.bounds[:,0],self.bounds[:,1])): #iterating over features and their respective bounds
                # checking i-th feature for smallest & largest value (belonging to an instance) that is falsely classified as positive:
                min_value_from_negative_instance = np.min(X[:,i][(y_instances==0)*(y_instances_hat==1)])
                max_value_from_negative_instance = np.max(X[:,i][(y_instances==0)*(y_instances_hat==1)])
                # set temp bound to be slightly above smallest negative instance value (so that instance is no longer classified as positive)
                temp_bound_min = np.nextafter(min_value_from_negative_instance, np.inf)
                temp_bound_max = np.nextafter(max_value_from_negative_instance, -np.inf)
                # calculate misclassification cost (difference in truly positive examples excluded before and after trying temp bound)
                positive_misclassification_cost_min = \
                np.sum(np.less(X[:,i][y_instances.astype(bool)], temp_bound_min)) - np.sum(np.less(X[:,i][y_instances.astype(bool)], min_bound))
                positive_misclassification_cost_max = \
                np.sum(np.greater(X[:,i][y_instances.astype(bool)], temp_bound_max)) - np.sum(np.greater(X[:,i][y_instances.astype(bool)], max_bound))
                # store misclassification cost and temp bound:
                misclassification_costs_min[i]=positive_misclassification_cost_min
                temp_bounds_min[i] = temp_bound_min
                misclassification_costs_max[i]=positive_misclassification_cost_max
                temp_bounds_max[i] = temp_bound_max
            
            # pick boundary adjustment with lowest misclassification cost:
            bound_index_min = np.argmin(misclassification_costs_min)
            min_misclassification_cost_min = misclassification_costs_min[bound_index_min]

            bound_index_max = np.argmin(misclassification_costs_max)
            min_misclassification_cost_max = misclassification_costs_max[bound_index_max]

            # Where exchange parameter comes into play: if a bound cannot be chosen without creating > lambda false negatives, then the program will break out of the loop and stop adjusting bounds
            if np.min((min_misclassification_cost_max, min_misclassification_cost_min)) > self.lamb:
                break
            # update only one bound:
            if min_misclassification_cost_min < min_misclassification_cost_max:
                self.bounds[bound_index_min, 0] = temp_bounds_min[bound_index_min]
                ### print(bound_index_min)
                ### print(temp_bounds_min[bound_index_min])
            else:
                self.bounds[bound_index_max, 1] = temp_bounds_max[bound_index_max]
                ### print(bound_index_max)
                ### print(temp_bounds_max[bound_index_max])
            
            y_instances_hat = self.predict_instances(X) # (now with a new bounds)

    def greedy_feature_selection(self, X, y_instances):
        super().greedy_feature_selection(X, y_instances)

    def predict_instances(self, X: np.ndarray) -> np.ndarray:
        return super().predict_instances(X)
    
    def predict(self, X: np.ndarray, bag_indices: List[np.ndarray]) -> np.ndarray:
        return super().predict(X, bag_indices)

    def lamb_estimation(self, X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray, labeled_instances = False, labeled_subset = False) -> float:
        print('Performing lamb estimation...')
        best_lamb = None
        best_acc = -np.inf
        for lamb_temporary in [1,2,3,4,5,6,7]:
            print(f'testing lamb value: {lamb_temporary}')
            classifier_temporary = BalancedAPRClassifier(lamb=lamb_temporary)
            accs = []
            splits = cv_split(bag_indices, y, folds=5, stratified=True)
            for bag_indices_train, y_train, bag_indices_test, y_test in splits:
                classifier_temporary.fit(X, bag_indices_train, y_train)
                y_test_hat = classifier_temporary.predict(X, bag_indices_test)
                acc = accuracy(y_test, y_test_hat)
                accs.append(acc)
            if np.average(accs) > best_acc:
                best_acc = np.average(accs)
                best_lamb = lamb_temporary
        return best_lamb