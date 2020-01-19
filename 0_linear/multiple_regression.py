# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#import data
file = pd.read_csv("endomondo.csv")
data = file.values
X = data[:, [0,1,2]]
Y = data[:, 3]*60 + data[:, 4]

#calculate weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_hat = np.dot(X, w.T)

print(w)

#calculate r squared
tmp = Y - Y_hat
tmp2 = Y - np.mean(Y)
r_squared = 1 - tmp.dot(tmp)/tmp2.dot(tmp2)
print("r squared = ", r_squared)

plt.scatter(X[:, 0], Y)
plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(X[:, 0], X[:, 1], Y)
#ax.plot(sorted(X[:, 0]), sorted(X[:, 1]), sorted(Y_hat))
#plt.show()