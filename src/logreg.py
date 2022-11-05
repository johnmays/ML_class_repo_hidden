import argparse
import numpy as np
import math
import util
import os
from numpy import linalg as LA
import matplotlib.pyplot as plt

from sting.classifier import Classifier
from sting.data import Feature, FeatureType, parse_c45


class LogReg(Classifier):
    def __init__(self, lamb, rate) -> None:
        self.lamb = lamb
        self.rate = rate
        
    def fit_s(self, X: np.ndarray, y: np.ndarray, epoch: int, research: bool = False, seed: int = 12345) -> None:
        if research == True:
            np.random.seed(seed) # to ensure the weights
        self.W = np.random.randn(len(X[0]),)+1
        self.B = np.random.randn()
        losses = np.zeros(epoch)
        for i in range(0, epoch):
            for example in range(0, len(y)):
                #X_temp = X[example].copy()
                gradient_w = self.gradient_w(X[example], y[example])
                gradient_b = self.gradient_b(X[example], y[example])
                #print(f"old W:{log.W}, old B:{log.B}")
                #print(f"gradient W:{gradient_w}, gradient B:{gradient_b}")
                self.W = self.W - (self.rate*gradient_w)
                self.B = self.B - (self.rate*gradient_b)
                #print(f"new W:{log.W}, new B:{log.B}")
            if research:
                losses[i] = self.cost(X,y)
        if research:
            return losses
        else:
            return None
        
    
    def fit(self, X: np.ndarray, y: np.ndarray, epoch: int, research: bool = False, seed: int = 12345) -> None:
        if research == True:
            np.random.seed(seed) # to ensure the weights
        self.W = np.random.randn(len(X[0]),)+1
        print(np.shape(self.W))
        self.B = np.random.randn()
        print(type(self.W))
        losses = np.zeros(epoch)
        for i in range(0, epoch):
            #X_temp = X[example].copy()
            gradient_w = self.gradient_w_r(X, y)
            gradient_b = self.gradient_b_r(X, y)
            #print(f"old W:{log.W}, old B:{log.B}")
            #print(f"gradient W:{gradient_w}, gradient B:{gradient_b}")
            self.W = self.W - (self.rate*gradient_w)
            self.B = self.B - (self.rate*gradient_b)
            #print(f"new W:{log.W}, new B:{log.B}")
            if research:
                losses[i] = self.cost(X,y)
        if research:
            return losses
        else:
            return None
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        W = self.W
        B = self.B
        prediction = np.sum(W * X, axis=1)+B
        for a in range(0, len(prediction)):
            if prediction[a] > 0:
                prediction[a] = 1
            else:
                prediction[a] = 0
        confidences = np.sum(W * X, axis=1)+B
        return prediction, confidences
    
    def sigmoid(self, X):
        W = self.W
        B = self.B
        #The logistic regression equation
        return 1/(1+np.exp(-(np.sum(W*X, axis=1)+B)))
    
    def gradient_w(self, X, y):
        W = self.W
        B = self.B
        lamb = self.lamb
        return X * (self.sigmoid(X)-y) + lamb * W
    
    def gradient_b(self, X, y):
        return np.sum(self.sigmoid(X)-y)

    def gradient_w_r(self, X, y):
        W = self.W
        B = self.B
        lamb = self.lamb
        return np.dot(X.T, (self.sigmoid(X)-y))/len(y) + lamb * W
    
    def gradient_b_r(self, X, y):
        W = self.W
        B = self.B
        return np.sum(self.sigmoid(X)-y)/len(y)
    
    def cost(self, X, y):
        W = self.W
        B = self.B
        lamb = self.lamb
        return np.sum(-y * np.log(1/(1+np.exp(-(np.sum(X*W, axis=1)+B)))) - (1-y) *np.log(1-(1/(1+np.exp(-(np.sum(X*W, axis=1)+B))))) + lamb/2*LA.norm(W)**2)


def evaluate_and_print_metrics(ys: np.ndarray, y_hats: np.ndarray, confidences: np.ndarray):
    """
    Print information about the performance of the naive Bayes classifier on testing data.

    Args:
        nb: the Logistic regression classifier to evaluate.
        X: An array of examples to use for evaluation.
        y: An array of class labels associated with the examples in X.
    """

    acc, precision, recall = [], [], []
    for i in range(len(ys)):
        acc.append(util.accuracy(ys[i], y_hats[i]))
        precision.append(util.precision(ys[i], y_hats[i]))
        recall.append(util.recall(ys[i], y_hats[i]))
    print(precision)

    all_ys, all_confidences = [], []
    for y in ys:
        all_ys.extend(y)
    for c in confidences:
        all_confidences.extend(c)
    
    print(f"{len(all_confidences)} ROC points")
    auc = util.auc(all_ys, all_confidences)

    print("\n***********\n* RESULTS *\n***********")
    print(f'Accuracy: {np.mean(acc):.3f} {np.var(acc):.3f}')
    print(f'Precision: {np.mean(precision):.3f} {np.var(precision):.3f}')
    print(f'Recall: {np.mean(recall):.3f} {np.var(recall):.3f}')
    print(f'AUR: {auc:.3f}\n')

def logreg(data_path: str, lamb: int, rate: int, use_cross_validation: bool = True, research: bool = False):
    # last entry in the data_path is the file base (name of the dataset)
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
    root_dir = os.sep.join(path[:-1])
    schema, X, y = parse_c45(file_base, root_dir)

    if use_cross_validation:
        datasets = util.cv_split(X, y, folds=5, stratified=False)
    else:
        datasets = ((X, y, X, y),)
    
    print("Fitting...")
    if not research:
        ys, y_hats, confidences = [], [], []
        for X_train, y_train, X_test, y_test in datasets:
            classifier = LogReg(lamb, rate)
            classifier.fit(X_train, y_train, 2500)

            y_hat, confidence = classifier.predict(X_test)
            ys.append(y_test)
            y_hats.append(y_hat)
            confidences.append(confidence)
            print("Evaluating...")
            evaluate_and_print_metrics(ys, y_hats, confidences)
    else: # Research Extension Fitting/Plotting:
        X_train, y_train, X_test, y_test = datasets[0]
        print(np.shape(X_train))
        print(np.shape(y_train))
        # Fitting two ways (stochastic GD, normal GD)
        seed = np.random.randint(10000,99999) # rand seed so the weights can still be random, but will be the same for all fit methods
        classifier_s = LogReg(lamb, rate)
        classifier_n = LogReg(lamb, rate)
        losses_stochastic = classifier_s.fit_s(X_train, y_train, 2500, research=True, seed=seed)
        losses_normal = classifier_n.fit(X_train, y_train, 2500, research=True, seed=seed)

        # Plotting the losses
        plt.plot(losses_stochastic, color="#b59fd0")
        plt.plot(losses_normal, color="#482980")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over time for SGD and GD")
        plt.show()

        # Evaluating for Stochastic GD
        y_hat, confidence = classifier_s.predict(X_test)
        print("Evaluating Stochastic...")
        evaluate_and_print_metrics((y_test,), (y_hat,), (confidences,))

        # Evaluating for Normal GD
        y_hat, confidence = classifier_n.predict(X_test)
        print("Evaluating Normal...")
        evaluate_and_print_metrics((y_test,), (y_hat,), (confidences,))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a Logistic regression algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('lamb', metavar='LAMBDA', type=float, help='Penalty parameter lambda.')
    parser.add_argument('rate', metavar='RATE', type=float, help='Learning rate.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.add_argument('--research', dest='research', action='store_true',
                        help='Enables the fit() and tree algo for the research question instead of running the normal fit/tree algo.')
    parser.set_defaults(cv=True, gain_ratio=False)
    args = parser.parse_args()

    data_path = os.path.expanduser(args.path)
    lamb = args.lamb
    rate = args.rate
    use_cross_validation = args.cv
    research = args.research

    if research and use_cross_validation:
        raise argparse.ArgumentError('The research cannot use CV.  Call research with the --no-cv flag')
    logreg(data_path, lamb, rate, use_cross_validation, research)