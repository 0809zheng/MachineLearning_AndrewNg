# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:06:37 2019

@author: zhijiezheng
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

K = 16 #clustering numbers
inner_iter = 1 #steps iteration
outer_iter = 1 #loss iteration

def min_index(x):
    temp_min = 1e12
    temp_index = 0
    for i in range(len(x)):
        if x[i] <= temp_min:
            temp_min = x[i]
            temp_index = i
    return temp_index

def cluster_assignment(x,miu):
    temp_c = np.zeros((len(x[:,0]),1))
    for i in range(len(x[:,0])):
        test = np.sum((x[i,:]-miu)**2,axis = 1)
        temp_c[i] = min_index(test.reshape(-1,1))
    return temp_c

def renew_centroid(x,c):
    count = np.zeros((K,1))
    temp_miu = np.zeros((K,len(x[0,:])))
    for i in range(len(x[:,0])):
        temp_miu[int(c[i])] += x[i]
        count[int(c[i])] += 1
    temp_miu /= count
    return temp_miu

def destortion(x,miu,c):
    loss =0.
    for i in range(len(x[:,0])):
        loss += np.sum((x[i]-miu[int(c[i])])**2)
    return loss

def random_init(x,K):
    miu = np.zeros((K,len(x[0,:])))
    rand_index = np.random.randint(0,len(x[:,0]),K)
    count = 0
    for i in rand_index:
        miu[count] = x[i]
        count += 1
    return miu

def draw_fig(x,miu,c):
    plt.scatter(miu[:,0],miu[:,1],color = 'red')
    for i in range(len(x[:,0])):
        if c[i] == 0:
            plt.scatter(x[i,0],x[i,1],color = 'black',alpha = 0.5)
        elif c[i] == 1:
            plt.scatter(x[i,0],x[i,1],color = 'blue',alpha = 0.5)
        elif c[i] == 2:
            plt.scatter(x[i,0],x[i,1],color = 'green',alpha = 0.5)
    plt.show()

def kmeans(str):
    data = loadmat(str)
    x = data['A'].reshape(-1,3)
    loss = 1e12
    for k in range(outer_iter):
        miu = random_init(x,K)
        for i in range(inner_iter):
            C = cluster_assignment(x,miu)
            miu = renew_centroid(x,C)
        temp_loss = destortion(x,miu,C)
        if temp_loss <= loss:
            loss = temp_loss
            final_miu = miu
            final_C = C
#    print('loss=%.2f'%loss)
#    draw_fig(x,final_miu,final_C)
    return x,final_miu,final_C