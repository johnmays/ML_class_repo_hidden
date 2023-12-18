import numpy as np

import argparse
import os.path
import sys

sys.path.append(".")
from group.data_io import load_data
from group import util

from apr import APRClassifier
from sbMIL import sbMILClassifier
from apr_extension import BalancedAPRClassifier

algo_list = ['apr', 'apr_extension', 'sbMIL']

if __name__ == '__main__':
    """
    Main method.

    Parses run arguments, imports/splits dataset(s), then calls algo(s) with formatted arguments and dataset(s).
    """

    # Parsing command line arguments:
    parser = argparse.ArgumentParser(description="Test John Mays's algos.")
    parser.add_argument('folder', metavar='folder', type=str, help='The folder that stores the datasets.')
    parser.add_argument('datasets', metavar='datasets', type=str, help='Comma separated list of dataset names.')
    parser.add_argument('-a', '--algo', type=str, help='Which algorithm to run.  If left blank, all will run on all three. Possible values are' + str(algo_list)[1:-1])
    parser.add_argument('--no-cv', dest='cv', action='store_false',
        help='Disables cross validation and trains on the full dataset.')
    parser.add_argument('--allpos', dest='allpos', action='store_true',
        help='for the APR algorithm, choose to run the all positive hypothesis.')
    parser.set_defaults(algo='all', cv=True, allpos= False)
    args = parser.parse_args()

    folder = os.path.expanduser(args.folder)
    algo = args.algo
    datasets = args.datasets.split(',')
    use_cross_validation = args.cv
    allpos = args.allpos

    if algo.casefold() != 'apr'.casefold() and allpos:
        raise argparse.ArgumentError('allpos is an attribute for APR only.  Leave unspecified for other algos.')

    algorithms = []
    if algo == 'all':
        print("All algorithms will run:")
        print('\n')
        algorithms = algo_list
    else:
        print(f"Only {algo} will run:")
        print('\n')
        algorithms = [algo]

    for dataset in datasets:
        # unpack:
        X, bag_indices, y = load_data(folder, dataset)
        print(f'Input data size: {X.shape} (num instances, num features)')
        print(f'Number of bags: {len(bag_indices)}')
        print('\n')
        if use_cross_validation:
            splits = util.cv_split(bag_indices, y, folds=5, stratified=True)
        else:
            splits = ((bag_indices, y, bag_indices, y),)
        
        for algorithm in algorithms:
            print(f"Running {algorithm} on {dataset}")
            classifier = None
            if algorithm.casefold() == 'apr'.casefold():
                if allpos:
                    classifier = APRClassifier(allpos=True)
                else:
                    classifier = APRClassifier()
            elif algorithm.casefold() == 'apr_extension'.casefold():
                classifier = BalancedAPRClassifier()
            elif algorithm.casefold() == 'sbMIL'.casefold():
                classifier = sbMILClassifier(C=0.1)
            else:
                argparse.ArgumentError('Algorithm was incorrectly specifed... Try one of these: ' + str(algo_list)[1:-1])

            accs = []
            precs = []
            recs = []
            i=1
            for bag_indices_train, y_train, bag_indices_test, y_test in splits:
                classifier.fit(X, bag_indices_train, y_train)

                y_hat = classifier.predict(X, bag_indices_test)
                acc = util.accuracy(y_test, y_hat)
                prec = util.precision(y_test, y_hat)
                rec = util.recall(y_test, y_hat)

                accs.append(acc)
                precs.append(prec)
                recs.append(rec)
                if use_cross_validation:
                    print(f"Fold {i}: Acc: {acc:.2f}, Prec: {prec:.2f}, Rec: {rec:.2f}")
                i+=1
            print('\n')
            print('Mean Performance:')
            print(f'Acc: {np.mean(accs):.2}, {np.std(accs):.2}')
            print(f'Prec: {np.mean(precs):.2}, {np.std(precs):.2}')
            print(f'Rec: {np.mean(recs):.2}, {np.std(recs):.2}')