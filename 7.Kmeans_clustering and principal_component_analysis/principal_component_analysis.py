# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:06:37 2019

@author: zhijiezheng
"""

import numpy as np
from PIL import Image
from scipy.io import loadmat

K = 100 #reductional dimension

def feature_scale(x):
    mean = np.sum(x,axis = 0)/len(x[:,0])
    sd = np.sum((x-mean)**2,axis = 0)/len(x[:,0])
    return (x-mean)/sd

def PCA(x,K):
    x_scale = feature_scale(x)
    cov = 1/len(x[:,0])*(x_scale.T).dot(x_scale)
    U,S,V = np.linalg.svd(cov)
    U_reduce = np.zeros((len(x[0,:]),K))
    U_reduce = U[:,0:K]
    x_reduce = x.dot(U_reduce)
    x_approx = x_reduce.dot(U_reduce.T)
    return x_reduce,x_approx

def draw_fig(x,x_approx,i = 1):
    im1 = Image.fromarray(gray(x[i].reshape(32,32)))
    im2 = Image.fromarray(gray(x_approx[i].reshape(32,32)))
    im1.show()
    im2.show()

def gray(f):
    f_new = (f-np.min(f))/(np.max(f)-np.min(f))*255
    return f_new

def main():
    data = loadmat('data/ex7faces.mat')
    x = data['X']    
    x_reduce,x_approx = PCA(x,K)
    draw_fig(x,x_approx)

main()