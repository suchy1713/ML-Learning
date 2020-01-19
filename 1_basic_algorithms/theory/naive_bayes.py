#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:41:16 2019

@author: suchy1713
"""

import numpy as np
from util import get_data
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

class Bayes:
    def fit(self, X, Y, smoothing=10e-3):
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        
        for c in labels:
            N, D = X.shape
            current_x = X[Y == c]
            self.gaussians[c] = {
                    'mean': current_x.mean(axis=0),
                    'cov': np.cov(current_x.T) + np.eye(D)*smoothing
                    }
            self.priors[c] = float(len(Y[Y == c]))/len(Y)
            
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)
    
    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        
        for c, g in self.gaussians.items():
            mean, cov = g['mean'], g['cov']
            P[:, c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[c])
        return np.argmax(P, axis=1)
    
X, Y = get_data(10000)
Ntrain = len(Y) // 2
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

model = Bayes()
t0 = datetime.now()
model.fit(Xtrain, Ytrain)
print("Training time:", (datetime.now() - t0))

t0 = datetime.now()
print("Train accuracy:", model.score(Xtrain, Ytrain))
print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

t0 = datetime.now()
print("Test accuracy:", model.score(Xtest, Ytest))
print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))