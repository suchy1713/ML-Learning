#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:40:43 2019

@author: suchy1713
"""

import numpy as np
from sortedcontainers import SortedList
from utils import classification_rate

class knn_model:    
    
    def __init__(self, X, Y, k):
        self.X, self.Y = X, Y
        self.k = k
    
    def perform_knn(self, X):        
        y = np.zeros(len(X))
        p = np.zeros(len(X))
        
        for i, x in enumerate(X): #loop over test/prediction shots
            sl = SortedList()
            
            for j, xt in enumerate(self.X): #loop over training shots
                diff = x - xt
                dist = diff.dot(diff)
                
                if len(sl) < self.k:
                    sl.add((dist, self.Y[j]))
                    
                else:
                    if dist < sl[-1][0]:
                        del sl[-1]
                        sl.add((dist, self.Y[j]))
            
            
            votes = {}
            
            for _,v in sl:
                votes[v] = votes.get(v, 0) + 1
                
            max_votes = 0
            max_votes_class = -1
            
            for v, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
                    
            y[i] = max_votes_class
            p[i] = votes.get(1, 0)/self.k
            
        return y, p
    
    
    def score(self, size):
        print("Measuring KNN performance... It will take a while.")
        y, p = self.perform_knn(self.X[:size, :])
        print("classification rate: ", classification_rate(y, self.Y[:size]))
       
    def predict(self, X):
        _, y = self.perform_knn(X)
        return y[0]
        
    def generate_X(self, l, bp, am, s, fb):
        X =  np.zeros((1, 27))
        X[0, l] = 1
        X[0, bp+14] = 1
        X[0, am+17] = 1
        X[0, s+22] = 1
        X[0, 26] = fb
        
        return X