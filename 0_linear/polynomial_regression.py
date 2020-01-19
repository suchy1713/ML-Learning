# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data
file = pd.read_csv("data_poly.csv", header=None)
file[2] = 1
file[3] = file.apply(lambda row: row[0]**2, axis=1)
print(file.head())
data = file.values
X = data[:, [2, 0, 3]]
Y = data[:, 1]

#calculate weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_hat = np.dot(X, w.T)

#calculate r squared
tmp = Y - Y_hat
tmp2 = Y - np.mean(Y)
r_squared = 1 - tmp.dot(tmp)/tmp2.dot(tmp2)
print("r squared = ", r_squared)

plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Y_hat))
plt.show()