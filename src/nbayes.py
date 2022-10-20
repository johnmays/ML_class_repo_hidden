import argparse
import numpy as np
from typing import List

from sting.classifier import Classifier
from sting.data import Feature, FeatureType, parse_c45

class NaiveBayes(Classifier):
    def __init__(self, schema: List[Feature], numbins: int, m: int) -> None:
        self.schema = schema
        self.num_bins = numbins
        self.ess = m
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


def evaluate_and_print_metrics(nbayes: NaiveBayes, X: np.ndarray, y: np.ndarray):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a decision tree algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('numbins', metavar='BINS', type=int,
                        help='The number of bins to create for any continuous attribute.')
    parser.add_argument('m', metavar='M', type=int,
                        help='The estimated equivalent sample size.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.set_defaults(cv=True, gain_ratio=False)
    args = parser.parse_args()