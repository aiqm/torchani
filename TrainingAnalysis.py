# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:07:34 2024

@author: Alex
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

training = pandas.read_csv("Training/training.log", index_col=0)
print(training.columns)
columns = training.columns
for col in columns:
    if "Train" not in col and "Test" not in col:
        continue

    if "Test" in col:
        plt.plot(training[col].dropna(), "--", label=col)
    if "Train" in col:
        plt.plot(training[col].dropna(), label=col)
        #plt.scatter(training[col].dropna().index, training[col].dropna(), marker=5) 
plt.legend()
plt.show()

for col in [x for x in columns if "Test" in x]:
    
    print(training[col])
    plt.scatter(training.index, training[col], s=1)
    plt.plot(moving_average(training[col].values, n=9))
    plt.title(col)
    plt.show()
    
    print((np.gradient(training[col])<0).sum()/training[col].shape[0])