from sting.classifier import Classifier
from sting.data import Feature, FeatureType, parse_c45
import os
from logreg import LogReg
import numpy as np
import util


# last entry in the data_path is the file base (name of the dataset)
path = os.path.expanduser("C:\\Users\\21995\\Desktop\\Computer Science\\CSDS 440\\Programming\\440data\\volcanoes").split(os.sep)
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
log = LogReg(0.01, W, B, 0.1)
costs = []
for i in range(0, 100):
    costs = np.append(costs, log.cost(X[0]))
    for example in range(0, len(y)):
        gradient_w = log.gradient_w(X[example])
        gradient_b = log.gradient_b(X[example])
        #log.W = log.W-(log.logistic_reg(X[example])-y[example])*log.rate*gradient_w
        #log.B = log.B-(log.logistic_reg(X[example])-y[example])*log.rate*gradient_b
        log.W = log.W-log.rate*gradient_w
        log.B = log.B-log.rate*gradient_b

#print(log.W)
#print(log.B)
print(costs)

print(util.accuracy(log.predict(X), y))