# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 21:42:00 2019


@author: zhijiezheng
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

learning_rate = 0.00001
lambd = 0.001
iter = 1

def cost(Y,R,Theta,X):
    cost = np.sum((Y-X.dot(Theta.T))*R**2)/2+lambd/2*(np.sum(Theta**2)+np.sum(X**2))
    return cost

def gradient(Y,R,Theta,X):
    gradTheta = np.dot(((Y-X.dot(Theta.T))*R).T,X) + lambd*Theta
    gradX = np.dot((Y-X.dot(Theta.T))*R,Theta) + lambd*X
    temp_Theta = Theta - learning_rate*gradTheta
    temp_X = X - learning_rate*gradX
    return temp_Theta,temp_X

def main():
    data = loadmat('data/ex8_movies.mat')
    Y = data['Y']
    R = data['R']
    nm = Y.shape[0]
    nu = Y.shape[1]
    assert Y.shape == (nm,nu)
    assert R.shape == (nm,nu)
    
    param = loadmat('data/ex8_movieparams.mat')
    Theta = param['Theta']
    X = param['X']
    k = X.shape[1]
    assert X.shape == (nm, k)
    assert Theta.shape == (nu,k)
    
    loss = []
    for i in range(iter):
        Theta,X = gradient(Y,R,Theta,X)
        loss.append(cost(Y,R,Theta,X))
    print(loss)
#    plt.figure()
#    plt.plot(np.arange(iter),loss)
#    plt.xlim(0,iter)
#    plt.show()

main()