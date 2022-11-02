import argparse
import numpy as np
import math
import util
import os
from numpy import linalg as LA

from sting.classifier import Classifier
from sting.data import Feature, FeatureType, parse_c45


class LogReg(Classifier):
    def __init__(self, lamb, rate) -> None:
        self.lamb = lamb
        self.rate = rate
        

    def fit(self, X: np.ndarray, y: np.ndarray, epoch: int) -> None:
        self.W = np.random.randn(len(X[0]),)+1
        self.B = np.random.randn()
        costs = []
        for i in range(0, epoch):
            costs.append(self.cost(X, y))
            for example in range(0, len(y)):
                #X_temp = X[example].copy()
                gradient_w = self.gradient_w(X[example], y[example])
                gradient_b = self.gradient_b(X[example], y[example])
                #print(f"old W:{log.W}, old B:{log.B}")
                #print(f"gradient W:{gradient_w}, gradient B:{gradient_b}")
                self.W = self.W - (self.rate*gradient_w)
                self.B = self.B - (self.rate*gradient_b)
                #print(f"new W:{log.W}, new B:{log.B}")
        return costs

    def predict(self, X: np.ndarray) -> np.ndarray:
        W = self.W
        B = self.B
        prediction = np.sum(W * X, axis=1)+B
        for a in range(0, len(prediction)):
            if prediction[a] > 0:
                prediction[a] = 1
            else:
                prediction[a] = 0
        return prediction
    
    def sigmoid(self, X):
        W = self.W
        B = self.B
        #The logistic regression equation
        return 1/(1+np.exp(-(np.sum(W*X)+B)))
    
    def gradient_w(self, X, y):
        W = self.W
        B = self.B
        lamb = self.lamb
        return X * (self.sigmoid(X)-y) + lamb * W
    
    def gradient_b(self, X, y):
        W = self.W
        B = self.B
        return (self.sigmoid(X)-y)
    
    def cost(self, X, y):
        W = self.W
        B = self.B
        lamb = self.lamb
        return np.sum(-y * np.log(1/(1+np.exp(-(np.sum(X*W, axis=1)+B)))) - (1-y) *np.log(1-(1/(1+np.exp(-(np.sum(X*W, axis=1)+B))))) + lamb/2*LA.norm(W)**2)


def evaluate_and_print_metrics(logreg: LogReg, X: np.ndarray, y: np.ndarray):
    costs = logreg.fit(X, y, 100)
    acc = util.accuracy(y, logreg.predict(X))
    print("\n***********\n* RESULTS *\n***********")
    print(f'Accuracy:{acc:.2f}')

def logreg(data_path: str, lamb: int, rate: int, use_cross_validation: bool = True):
    # last entry in the data_path is the file base (name of the dataset)
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
    root_dir = os.sep.join(path[:-1])
    schema, X, y = parse_c45(file_base, root_dir)

    if use_cross_validation:
        datasets = util.cv_split(X, y, folds=5, stratified=True)
    else:
        datasets = ((X, y, X, y),)
    
    for X_train, y_train, X_test, y_test in datasets:
        classifier = LogReg(lamb, rate)
        classifier.fit(X_train, y_train, 100)
        evaluate_and_print_metrics(classifier, X_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a Logistic regression algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('lamb', metavar='LAMBDA', type=float, help='The weight decay coefficient.')
    parser.add_argument('rate', metavar='RATE', type=float, help='The weight decay coefficient.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.set_defaults(cv=True, gain_ratio=False)
    args = parser.parse_args()

    data_path = os.path.expanduser(args.path)
    lamb = args.lamb
    rate = args.rate
    use_cross_validation = args.cv

    logreg(data_path, lamb, rate, use_cross_validation)