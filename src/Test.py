from sting.classifier import Classifier
from sting.data import Feature, FeatureType, parse_c45
import os
from logreg import LogReg
import numpy as np


# last entry in the data_path is the file base (name of the dataset)
path = os.path.expanduser("C:\\Users\\21995\\Desktop\\Computer Science\\CSDS 440\\Programming\\440data\\voting").split(os.sep)
file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
root_dir = os.sep.join(path[:-1])
schema, X, y = parse_c45(file_base, root_dir)
"""for a in schema:
    if a.ftype == FeatureType.NOMINAL:
        print(a.nominal_values)
print(X[0])"""
log = LogReg(0.1)
print(len(X[0]))
print(np.shape(X[0]))
W = np.random.randn(11,)
B = np.random.randn()
print(log.logistic_reg(W, X[0], B))