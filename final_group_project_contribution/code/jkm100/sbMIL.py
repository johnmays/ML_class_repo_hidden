"""
Created on 12/04/22 by John Mays

This is my implementation of the balanced SVM algorithm (sbMIL) for the problem of sparse multiple-instance classification, originally described in Bunescu & Mooney's 2007 paper, "Multiple Instance Learning for Sparse Positive Bags."
"""
import sys
sys.path.append(".")

from group.classifier import MILClassifier
from group.util import accuracy, cv_split, auc
from typing import List, Optional, Tuple

import numpy as np
import cvxpy as cp # optimization package
import random

random.seed(0)

class sbMILClassifier(MILClassifier):
    def __init__(self, C, eta= None):
        self.w = None # weights
        self.b = None # bias

        self.C = C # SVM capacity parameter must be specified
        self.eta = eta # an eta may be specified upon instantiation
        self.eta_was_set = True
        if self.eta is None:
            self.eta_was_set = False

    def fit(self, X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray):
        # go through eta estimation if eta is not specified:
        if self.eta_was_set == False:
            self.eta = self.eta_estimation(X, bag_indices, y, labeled_instances = False)
            print(f'A value of {self.eta} was chosen by eta estimation...')
        # then optimize weights:
        self.solve_sbMIL(X, bag_indices, y)

    def solve_sbMIL(self, X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray):
        # instantiating weights & bias:
        self.w = np.zeros(np.shape(X)[1])
        self.b = 0
        # first update W,B based on assumption that bags are sparse:
        self.solve_sMIL(X, bag_indices, y)
        # split X matrix by true bag label:
        X_from_pos_bags, X_from_neg_bags = self.split_instance_matrix_by_label(X, bag_indices, y)
        # order and label instances from postive bags by taking w phi(x) + b as a score and picking the top eta*(#instances) of them
        # note: these predictions will be in the +/- 1 format required by SIL SVMs:
        X_from_pos_bags, y_from_pos_bags = self.estimate_positive_instances(X_from_pos_bags)
        X_after_estimation = np.vstack((X_from_pos_bags, X_from_neg_bags))
        y_after_estimation = np.concatenate((y_from_pos_bags, -np.ones(np.shape(X_from_neg_bags)[0])))
        self.solve_SIL(X_after_estimation, y_after_estimation)
        # w,b should now be fully fitted!

    def solve_sMIL(self, X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray):
        # unpack X for problem (neg bags go into instance matrix, pos bags go into sum of their instances by bag matrix)
        phi_X_pos_bags, pos_bag_lengths, phi_X_neg_instances = self.split_instance_matrix_for_sMIL(X, bag_indices, y)
        num_neg_instances = np.shape(phi_X_neg_instances)[0]
        num_pos_bags = np.shape(phi_X_pos_bags)[0]

        # initializing optimization variables:
        w = cp.Variable(np.shape(X)[1])
        b = cp.Variable()
        C = self.C
        slack_neg_instances = cp.Variable(num_neg_instances)
        slack_pos_bags = cp.Variable(num_pos_bags)

        # describing objective function:
        regularity = cp.norm(w, 2)
        total_expression = 0.5*(regularity**2) + (C/num_neg_instances)*cp.sum(slack_neg_instances) + (C/num_pos_bags)*cp.sum(slack_pos_bags)
        objective = cp.Minimize(total_expression)

        # describing constraints:
        constraints = []
        constraints.append(phi_X_neg_instances@w+b <= -1 + slack_neg_instances)
        constraints.append((phi_X_pos_bags@w)/pos_bag_lengths+b >= (2-pos_bag_lengths)/pos_bag_lengths - slack_pos_bags)
        constraints.append(slack_neg_instances >= 0)
        constraints.append(slack_pos_bags >= 0)
        
        # performing minimization:
        problem = cp.Problem(objective, constraints)
        problem.solve()
        self.w = w.value
        self.b = b.value

    def solve_SIL(self, X: np.ndarray, y: np.ndarray):
        """
        Solves SIL SVM (single-instance learning) like a regular old SVM problem.

        input y is by instances
        """
        phi_X = X

        # initializing optimization variables:
        w = cp.Variable(np.shape(phi_X)[1])
        b = cp.Variable()
        C = self.C
        num_instances = np.shape(X)[0]
        slack = cp.Variable(num_instances)

        # describing objective function:
        regularity = cp.norm(w, 2)
        total_expression = 0.5*(regularity**2) + (C/num_instances)*cp.sum(slack)
        objective = cp.Minimize(total_expression)

        # describing constraints:
        constraints = []
        constraints.append(cp.multiply(y, (phi_X@w+b)) + slack >= 1)
        constraints.append(slack >= 0)
        
        # performing minimization:
        problem = cp.Problem(objective, constraints)
        problem.solve()
        self.w = w.value
        self.b = b.value

    def estimate_positive_instances(self, X_from_pos_bags) -> np.ndarray:
        """
        This takes the set of all instances from true positive bags, orders them by w*phi(x)+b, and labels the top top eta*(number of instances) of them as positive, and returns both the reordered X(positive) matrix and the labels in the y matrix.
        """
        # score X from positive bags according to estimated eta parameter:
        scores = np.zeros(np.shape(X_from_pos_bags)[0])
        for i, x in enumerate(X_from_pos_bags):
            scores[i] = np.dot(self.w,x)+self.b
        # append scores to first column of X:
        scores_X = np.hstack((scores.reshape((np.size(scores),1)),X_from_pos_bags))
        # sort rows (instances) by scores (highest to lowest):
        scores_X = scores_X[scores_X[:, 0].argsort()[::-1]]
        # taking the first column (scores) back off:
        X_from_pos_bags = scores_X[:,1:]
        y_hat_instances = -np.ones(np.shape(X_from_pos_bags)[0])
        last_positive_index = int(np.floor(self.eta*np.size(y_hat_instances)))
        y_hat_instances[0:last_positive_index] = 1
        return X_from_pos_bags, y_hat_instances

    def split_bag_indices_by_label(self, bag_indices: List[np.ndarray], y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        helper
        """
        bag_indices_pos, bag_indices_neg = [], []
        for indices, label in zip(bag_indices, y):
            if label == 1:
                bag_indices_pos.append(indices) 
            else:
                bag_indices_neg.append(indices) 
        return bag_indices_pos, bag_indices_neg

    def split_instance_matrix_for_sMIL(self, X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Should return a matrix of phi(X) (by bag), for the positively labeled bags, a list of positive bag lengths, and a matrix of phi(x) (by instance) for negatively labeled bags.  Exclusively needed for the solve_sMIL subproblem.
        """
        phi_X_pos_bags, phi_X_neg_instances = None, None
        pos_bag_lengths = []
        for indices, label in zip(bag_indices, y):
            if label == 1:
                phi_X = 0
                bag_length = 0
                for index in indices:
                    x = X[index]
                    phi_X += x
                    bag_length += 1
                if phi_X_pos_bags is None:
                    phi_X_pos_bags = np.copy(phi_X)
                else:
                    phi_X_pos_bags = np.vstack((phi_X_pos_bags, phi_X))
                pos_bag_lengths.append(bag_length)
            else:
                for index in indices:
                    x = X[index]
                    if phi_X_neg_instances is None:
                        phi_X_neg_instances = np.copy(x)
                    else:
                        phi_X_neg_instances = np.vstack((phi_X_neg_instances, x))
                
        return phi_X_pos_bags, np.array(pos_bag_lengths), phi_X_neg_instances
    
    def split_instance_matrix_by_label(self, X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        helper
        """
        X_pos_bags, X_neg_bags = None, None
        for indices, label in zip(bag_indices, y):
            for index in indices:
                x = X[index]
                if label == 1:
                    if X_pos_bags is None:
                        X_pos_bags = np.copy(x)
                    else:
                        X_pos_bags = np.vstack((X_pos_bags, x))
                else:
                    if X_neg_bags is None:
                        X_neg_bags = np.copy(x)
                    else:
                        X_neg_bags = np.vstack((X_neg_bags, x))
        return X_pos_bags, X_neg_bags

    def predict_instances(self, X: np.ndarray, one_zero = True) -> np.ndarray:
        """
        helper
        """
        y_hat = np.zeros(np.shape(X)[0])
        for i, x in enumerate(X):
            predicted_label = np.sign(np.dot(self.w,x)+self.b)
            if predicted_label == 0:
                y_hat[i] = 1
            else:
                y_hat[i] = predicted_label
        assert np.all((y_hat==1) + (y_hat==-1)) # all labels should be -1 or 1 at this point
        if one_zero:
            y_hat[y_hat==-1] = 0 # make all of the -1s into 0s.
        return y_hat
    
    def predict(self, X: np.ndarray, bag_indices: List[np.ndarray], one_zero = True) -> np.ndarray:
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
        y_instances_hat = self.predict_instances(X, one_zero=one_zero)

        # initializing predictions vector:
        y_hat = None
        if one_zero:
            y_hat = np.zeros(len(bag_indices))
        else:
            y_hat = -np.ones(len(bag_indices))

        # using instance predictions to make bag predictions using SMI:
        for bag_index, indices in enumerate(bag_indices):
            for index in indices:
                if y_instances_hat[index] == 1:
                    y_hat[bag_index] = 1
        return y_hat

    def get_confidences(self, X: np.ndarray, bag_indices: List[np.ndarray]) -> np.ndarray:
        """
        Returns prediction confidences as dicated by Bunescu & Mooney, 2007:
        'The prediction confidence for each bag is computed as the maximum over the prediction confidence of each instance in the bag. At instance level, the confidence is set to the value of the decision function f (x) = w phi(x) + b on that instance.'
        """
        conf_instances = np.zeros(np.shape(X)[0])
        for i, x in enumerate(X):
            conf_instances[i] = np.dot(self.w,x)+self.b
        conf_bags = -np.inf * np.ones(len(bag_indices))
        for bag_index, indices in enumerate(bag_indices):
            for index in indices:
                if conf_instances[index] > conf_bags[bag_index]:
                    conf_bags[bag_index] = conf_instances[index]
        return conf_bags


    def eta_estimation(self, X: np.ndarray, bag_indices: List[np.ndarray], y: np.ndarray, labeled_instances = False, labeled_subset = False) -> float:
        """
        Runs cross validation with set of test values for hyperparameter eta.  see pseudocode for more.
        """
        print('Performing eta estimation...')
        if labeled_instances: # if there are ground truth instance labels
            # then y should be instance labels not bag labels:
            assert np.size(y) == np.shape(X)[0]
            if labeled_subset: # we were only given an instance-labeled subset of the training data L
                return np.average(y)
            else: # the entire training set X is instance-labeled
                sampled_labels = []
                for i in range(np.size(y)):
                    if np.random.uniform(low=0.0,high=1.0) < 0.1:
                        sampled_labels.append(y[i])
                return np.average(sampled_labels)
        else:
            # If no truly labeled instances are available, we must find an optimal eta
            best_eta = None
            best_auc = -np.inf
            for eta_temporary in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                print(f'testing eta value: {eta_temporary}')
                classifier_temporary = sbMILClassifier(C=self.C, eta=eta_temporary)
                aucs = []
                splits = cv_split(bag_indices, y, folds=9, stratified=True)
                for bag_indices_train, y_train, bag_indices_test, y_test in splits:
                    classifier_temporary.fit(X, bag_indices_train, y_train)
                    p_y_hat = classifier_temporary.get_confidences(X, bag_indices_test)
                    area_under_curve = auc(y_test, p_y_hat)
                    aucs.append(area_under_curve)
                if np.average(aucs) > best_auc:
                    best_auc = np.average(aucs)
                    best_eta = eta_temporary
            return best_eta