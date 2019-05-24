# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 08:24:03 2019

@author: zhijiezheng
"""

import numpy as np
from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt
import time

learning_rate = 0.1 #设定学习率
iterations = 10000 #设定梯度下降轮数
regularization_parameter = 0.001 #正则化系数
K = 10 #设置分类数

#定义函数实现文件输入
def read_xy():
    data = loadmat('ex3data1.mat')
    x = data['X']
    y = data['y']%10
    x = np.c_[np.ones(len(y)),x]
    return x,y

#随机选取图片展示
def display_data(x,k):
    im = Image.fromarray((x[k,1:]*255).reshape(20,20))
    plt.imshow(im)
    
#定义假设函数
def h(theta,x):
    return x.dot(theta)

def sigmoid(theta,x):
    return 1./(1+np.exp(-h(theta,x)))

#定义损失函数
def cost(theta,x,y):
    return -np.sum(y*np.log(sigmoid(theta,x))+(1-y)*np.log(1-sigmoid(theta,x)))/len(y)+np.sum(pow(theta,2))/(2*len(y))

#实现批量梯度下降算法
def batch_gardient_descent(theta,x,y):
    return theta-learning_rate*(x.T.dot((sigmoid(theta,x)-y))+
                               regularization_parameter*(np.r_[[[0]],theta[1:]].reshape(-1,1)))/len(y)
    
def train(theta,x,y):
    for i in range(K):
        y_ = np.array([1 if l==i else 0 for l in y]).reshape(-1,1)
        theta_ = np.r_[[[0]],theta[1:,0].reshape(-1,1)]
        for j in range(iterations):
            theta_ = batch_gardient_descent(theta_,x,y_)
        theta[0:,i] = theta_.reshape(-1)
    return theta

def test(theta,x,y):
    count = 0
    predict = sigmoid(theta,x)//np.max(sigmoid(theta,x),axis=1).reshape(-1,1)
    for i in range(len(y)):
        if predict[i,y[i]]==1.:
            count+=1
    print('最终的准确率是：{}%'.format(count/len(y)*100))
        

def main():
    t1 = time.perf_counter()
    x,y = read_xy()
    display_data(x,0) #展示第0张图片
    theta = np.ones((len(x[0]),K)) #初始化参数
    theta = train(theta,x,y)
    test(theta,x,y)
    t2 = time.perf_counter()
    print('程序总用时：{:.3f}s'.format(t2-t1))

#如果main是主函数，则执行main    
if __name__ == '__main__':
    main()