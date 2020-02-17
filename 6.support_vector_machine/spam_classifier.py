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
data = loadmat('data/spamTrain.mat')
X = data['X']
y = data['y']

data_test = loadmat('data/spamTest.mat')
X_test = data_test['Xtest']
y_test = data_test['ytest']

model = SVC(C = 1)
model.fit(X,y)
y_ = model.predict(X)

print('Training Accuracy:%.3f'%model.score(X,y_))
print('Test Accuracy:%.3f'%model.score(X_test,y_test))