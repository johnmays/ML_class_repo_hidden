import argparse
import numpy as np
import math
from numpy import linalg as LA

from sting.classifier import Classifier
from sting.data import Feature, FeatureType, parse_c45


class LogReg(Classifier):
    def __init__(self, lamb, W, B, rate) -> None:
        self.lamb = lamb
        self.W = W
        self.B = B
        self.rate = rate
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

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
    
    def gradient_w(self, X: np.ndarray, y):
        W = self.W
        B = self.B
        lamb = self.lamb
        #The partial derivative of the function respect to W 
        return y*(X*np.exp(-(np.sum(W*X)+B)))/(1+np.exp(-(np.sum(W*X)+B)))-(1-y)*X/(1+np.exp(-(np.sum(W*X)+B)))+lamb*W
    
    def gradient_b(self, X: np.ndarray, y):
        W = self.W
        B = self.B
        #The partial derivative of the function respect to B
        return y*np.exp(-(np.sum(W*X)+B))/(1+np.exp(-(np.sum(W*X)+B)))-(1-y)/(1+np.exp(-(np.sum(W*X)+B)))
    
    def logistic_reg(self, X: np.ndarray):
        W = self.W
        B = self.B
        #The logistic regression equation
        return 1/(1+np.exp(-(np.sum(W*X)+B)))
    
    def cost(self, X, y):
        W = self.W
        B = self.B
        lamb = self.lamb
        return np.sum(-y * np.log(1/(1+np.exp(-(np.sum(X*W, axis=1)+B)))) - (1-y) *np.log(1-(1/(1+np.exp(-(np.sum(X*W, axis=1)))))) + lamb/2*LA.norm(W)**2)

    def update_W(self, gradient_w):
        self.W = self.W - gradient_w
    
    def update_B(self, gradient_B):
        self.B = self.B - gradient_B


def evaluate_and_print_metrics(logreg: LogReg, X: np.ndarray, y: np.ndarray):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a Logistic regression algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('lamb', metavar='LAMBDA', type=float,
                        help='The weight decay coefficient.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.set_defaults(cv=True, gain_ratio=False)
    args = parser.parse_args()