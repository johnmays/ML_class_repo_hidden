import argparse
import os.path
import numpy as np
from typing import List, Tuple

import util

from sting.classifier import Classifier
from sting.data import Feature, FeatureType, parse_c45

class NaiveBayes(Classifier):
    def __init__(self, schema: List[Feature], num_bins: int, m: int) -> None:
        self.schema = schema
        self.num_bins = num_bins
        self.ess = m

        self.prior = 0
        self.model = []
        # The ith index of self.model contains conditional probabilities
        # associated with the ith attribute of schema.
        for feature in schema:
            if feature.ftype == FeatureType.NOMINAL:
                self.model.append(np.zeros((2, len(feature.values))))
            else:
                self.model.append(np.zeros((2, self.num_bins)))
        self.ranges = []


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        num_ones = sum(y)
        num_zeros = len(y) - num_ones
        self.prior = num_ones / len(y)

        # Calculate smoothed conditional probabilities
        for index in range(len(self.schema)):
            print(f"Fitting feature {index+1}/{len(self.schema)}", end='\r')
            feature = self.schema[index]
            parameters = self.model[index]
            examples = np.array(X[:, index])
            if feature.ftype == FeatureType.CONTINUOUS:
                V = self.num_bins
                # Record ranges of continuous attributes
                minimum = np.min(examples)
                maximum = np.max(examples)
                bin_width = (maximum - minimum) / self.num_bins
                self.ranges.append((minimum, maximum, bin_width))
                # Vectorize conversion to bin indices
                binned = np.floor((examples - minimum) / bin_width)
            else:
                V = len(feature.values)
                self.ranges.append(None)
                # Convert examples to bin indices
                binned = examples - 1
            # Gather counts
            for j in range(len(binned)):
                bin = min(binned[j], V - 1)
                parameters[int(y[j])][int(bin)] += 1
            # Smoothing
            for b in range(V):
                parameters[0][b] = (parameters[0][b] + (self.ess / V)) / (num_zeros + self.ess)
                parameters[1][b] = (parameters[1][b] + (self.ess / V)) / (num_ones + self.ess)
        print("\nDone with fold!")
    

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method uses the Naive Bayes classifier to label a set of examples, X.

        Args:
            X: Testing data of shape (n_examples, n_features).

        Returns: Predictions of shape (n_examples,), either 0 or 1
        """
        y = []
        confidences = []
        for example in X:
            # 'likelihood' tracks the probability of X, given 0 and 1 respectively
            likelihood = [1 - self.prior, self.prior]
            for index in range(len(self.schema)):
                feature = self.schema[index]
                parameters = self.model[index]
                if feature.ftype == FeatureType.CONTINUOUS:
                    r = self.ranges[index]
                    bin = int(min((example[index] - r[0]) / r[2], self.num_bins - 1))
                    likelihood[0] *= parameters[0][bin]
                    likelihood[1] *= parameters[1][bin]
                else:
                    bin = int(example[index] - 1)
                    likelihood[0] *= parameters[0][bin]
                    likelihood[1] *= parameters[1][bin]

            y.append((likelihood[1] >= likelihood[0]) * 1)
            # confidence = P(Y = y | X)
            # We've already calculated P(Y = y, X) for all two values of Y.
            # Therefore, we can find P(X) through marginalization (sum(likelihood)),
            # and can easily find the confidence through Bayes' rule.
            p_y_given_x = likelihood[y[-1]] / sum(likelihood)
            confidences.append(p_y_given_x if y[-1] else 1 - p_y_given_x)

        return y, confidences


def evaluate_and_print_metrics(ys: np.ndarray, y_hats: np.ndarray, confidences: np.ndarray):
    """
    Print information about the performance of the naive Bayes classifier on testing data.

    Args:
        nb: the naive Bayes classifier to evaluate.
        X: An array of examples to use for evaluation.
        y: An array of class labels associated with the examples in X.
    """

    acc, precision, recall = [], [], []
    for i in range(len(ys)):
        acc.append(util.accuracy(ys[i], y_hats[i]))
        precision.append(util.precision(ys[i], y_hats[i]))
        recall.append(util.recall(ys[i], y_hats[i]))

    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)

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


def nbayes(data_path: str, num_bins: int, m: int, use_cross_validation: bool = True):
    # last entry in the data_path is the file base (name of the dataset)
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
    root_dir = os.sep.join(path[:-1])
    schema, X, y = parse_c45(file_base, root_dir)

    if use_cross_validation:
        datasets = util.cv_split(X, y, folds=5, stratified=True)
    else:
        datasets = ((X, y, X, y),)
    
    ys, y_hats, confidences = [], [], []
    for X_train, y_train, X_test, y_test in datasets:
        classifier = NaiveBayes(schema, num_bins, m)
        classifier.fit(X_train, y_train)

        y_hat, confidence = classifier.predict(X_test)
        ys.append(y_test)
        y_hats.append(y_hat)
        confidences.append(confidence)
    
    print("Evaluating...")
    evaluate_and_print_metrics(ys, y_hats, confidences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a Naive Bayse algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('numbins', metavar='BINS', type=int,
                        help='The number of bins to create for any continuous attribute.')
    parser.add_argument('m', metavar='M', type=float,
                        help='The estimated equivalent sample size.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.set_defaults(cv=True, gain_ratio=False)
    args = parser.parse_args()

    data_path = os.path.expanduser(args.path)
    num_bins = args.numbins
    m = args.m
    use_cross_validation = args.cv

    nbayes(data_path, num_bins, m, use_cross_validation)

