#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:16:53 2019

@author: suchy1713
"""

from data_preprocessing import get_binary_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def forward(X, w, b):
    return sigmoid(X.dot(w) + b)

def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy_error(T, Y):
    return -np.mean(T*np.log(Y) + (1-T)*np.log(1-Y))

X, Y = get_binary_data()
X, Y = shuffle(X, Y)

Xtrain = X[:-100]
Ytrain = Y[:-100]
Xtest = X[-100:]
Ytest = Y[-100:]

D = X.shape[1]
w = np.random.randn(D)
b = 0

train_costs = []
test_costs = []
learning_rate = 0.001
epochs = 10000

#main training loop
for i in range(epochs):
    pYtrain = forward(Xtrain, w, b)
    pYtest = forward(Xtest, w, b)
    
    ctrain = cross_entropy_error(Ytrain, pYtrain)
    ctest = cross_entropy_error(Ytest, pYtest)
    
    train_costs.append(ctrain)
    test_costs.append(ctest)
    
    w -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate*(pYtrain - Ytrain).sum()
    
    if i%1000 == 0:
        print(i, ctrain, ctest)
    
print("Final train score = ", classification_rate(Ytrain, np.round(pYtrain)))
print("Final test score = ", classification_rate(Ytest, np.round(pYtest)))

legend1, = plt.plot(train_costs, label="train cost")
legend2, = plt.plot(test_costs, label="test cost")
plt.legend([legend1, legend2])
plt.show()