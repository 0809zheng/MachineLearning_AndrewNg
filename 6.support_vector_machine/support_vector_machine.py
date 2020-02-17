# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 21:42:00 2019

@author: zhijiezheng
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC

plt.figure()
data = loadmat('data/ex6data3.mat')
X = data['X']
y = data['y']
m = len(y)
for i in range(m):
    plt.scatter(X[i,0],X[i,1],color = 'green' if y[i] else 'red',marker = 'x' if y[i] else 'o')

model = SVC(C =100, kernel='rbf', degree = 3)
model.fit(X,y)

for i in np.linspace(-0.7,0.4,70):
    for j in np.linspace(-.7,.6,70):
        plt.scatter(i,j,color = 'white' if model.predict(np.array([i,j]).reshape(1,2)) else 'black',alpha = .2)
        
plt.xlim(-0.7,0.4)
plt.ylim(-.7,.6)
plt.show()