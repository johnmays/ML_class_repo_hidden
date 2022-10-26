import argparse
import numpy as np
import math

from sting.classifier import Classifier
from sting.data import Feature, FeatureType, parse_c45

class LogReg(Classifier):
    def __init__(self, lamb) -> None:
        self.lamb = lamb
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def gradient_w(self, W: np.ndarray, X: np.ndarray, B: np.ndarray):
        #The partial derivative of the function respect to W
        return X/(1+math.e**(-W*X+B))+self.lamb*W
    
    def gradient_b(self, W: np.ndarray, X: np.ndarray, B: np.ndarray):
        #The partial derivative of the function respect to B
        return -1/(1+math.e**(-W*X+B))
    
    def logistic_reg(self, W: np.ndarray, X: np.ndarray, B: np.ndarray):
        #The logistic regression equation
        return 1/(1+math.e**(-(np.sum(W*X)+B)))


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