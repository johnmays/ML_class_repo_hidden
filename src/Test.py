from sting.classifier import Classifier
from sting.data import Feature, FeatureType, parse_c45
import os
from logreg_rethink import LogReg
import numpy as np
import util
from sklearn.linear_model import LogisticRegression


# last entry in the data_path is the file base (name of the dataset)
path = os.path.expanduser("C:\\Users\\21995\\Desktop\\Computer Science\\CSDS 440\\Programming\\440data\\spam").split(os.sep)
file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
root_dir = os.sep.join(path[:-1])
schema, X, y = parse_c45(file_base, root_dir)
"""for a in schema:
    if a.ftype == FeatureType.NOMINAL:
        print(a.nominal_values)
print(X[0])"""
np.random.seed(300)

W = np.random.randn(len(X[0]),)+1
B = np.random.randn()
log = LogReg(0.1, W, B, 0.01)
print(W)
print(B)
"""costs = []
for i in range(0, 100):
    costs.append(log.cost(X, y))
    #X_temp = X[example].copy()
    gradient_w = log.dw(X, y)
    gradient_b = log.db(X, y)
    #print(f"old W:{log.W}, old B:{log.B}")
    #print(f"gradient W:{gradient_w}, gradient B:{gradient_b}")
    log.W = log.W - (log.rate*gradient_w)
    #log.B = log.B - (log.rate*gradient_b)
    #print(f"new W:{log.W}, new B:{log.B}")"""

costs = log.fit(X, y, 20)

print(log.W)
print(log.B)
print(costs)

print(util.accuracy(log.predict(X), y))
"""clf = LogisticRegression(random_state=0).fit(X, y)
print(clf.score(X, y))"""