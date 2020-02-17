# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 21:42:00 2019

@author: zhijiezheng
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def anomaly_detection(x):
    m = len(x[:,0])
    n = len(x[0,:])
    coef = np.zeros((n,2))
    for i in range(n):
        #mean
        coef[i,0] = np.sum(x[:,i])/m
        #standard deviation
        coef[i,1] = np.sum((x[:,i]-coef[i,0])**2)/m
    return coef

def if_normal(sample,coef):
    x = sample.reshape(1,-1)
    n = len(x[0,:])
    p = 1.
    for i in range(n):
        p *= 1/((2*np.pi)**0.5*coef[i,1])*np.e**(-(x[:,i]-coef[i,0])**2/(2*coef[i,1]**2))
    return p

def val(x_val,y_val,coef,epsilon=0.5):
    m_val = len(y_val)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(m_val):
        p = if_normal(x_val[i],coef)
        if (p >= epsilon) and (y_val[i] == 0):
            TN += 1
        elif (p >= epsilon) and (y_val[i] == 1):
            FN += 1
        elif (p < epsilon) and (y_val[i] == 0):
            FP += 1
        elif (p < epsilon) and (y_val[i] == 1):
            TP += 1
    precision = TP/(TP+FP+1e-32)+1e-32
    recall = TP/(TP+FN+1e-32)+1e-32
    F1_score = 2*precision*recall/(precision+recall)
    return F1_score

def main():
    plt.figure()
    data = loadmat('data/ex8data1.mat')
    x = data['X']
    x_val = data['Xval']
    y_val = data['yval']
    
    plt.figure()
    plt.scatter(x[:,0],x[:,1])
    plt.show()
    
    coef = anomaly_detection(x)
    ep = np.linspace(0,0.01,1000)
    F1_score = []
    for i in ep:
        F1_score.append(val(x_val,y_val,coef,i))
    plt.figure()
    plt.plot(ep,F1_score)
    plt.show()

    plt.figure()
    for i in range(len(x_val[:,0])):
        plt.scatter(x_val[i,0],x_val[i,1],color = 'red' if if_normal(x_val[i],coef)<0.002 else 'blue')
    plt.show()

main()