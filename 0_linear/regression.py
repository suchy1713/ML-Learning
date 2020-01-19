#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:45:07 2019

@author: suchy1713
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import data
file = pd.read_csv("data_1d.csv", header=None)
data = file.values

#divide data
X = data[:, 0]
Y = data[:, 1]

#calculate a and b
denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean()*X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator

#get best fitting line
Y_hat = a*X + b

#calculate r squared
tmp1 = Y - Y_hat
tmp2 = Y - np.mean(Y)
r_squared = 1 - tmp1.dot(tmp1)/tmp2.dot(tmp2)
print("R squared = ", r_squared)

#plot
plt.scatter(X,Y)
plt.plot(X, Y_hat)
plt.show()