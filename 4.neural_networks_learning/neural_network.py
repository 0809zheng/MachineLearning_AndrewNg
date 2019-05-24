# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:27:48 2019

@author: zhijiezheng
"""

import numpy as np
from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt
import time

HIDDEN_NODE = 25 #设置隐藏层神经元的个数
learning_rate = 0.1 #设定学习率
iterations = 10000 #设定梯度下降轮数
regularization_parameter = 0.1 #正则化系数
K = 10 #设置分类数

#定义函数实现文件输入
def read_xy():
    data = loadmat('ex4data1.mat')
    x = data['X']
    y = data['y']%10
    x = np.c_[np.ones(len(y)),x]
    return x,y

def sigmoid(x):
    return 1./(1+np.exp(-x))

#定义损失函数
def cost(theta1,theta2,x,y_):
    J1,J2 = 0,0
    y_predict,A2 = fordward_propagation(theta1,theta2,x)
    J1 = -np.sum(y_*np.log(y_predict))-np.sum((1-y_)*np.log(1-y_predict))
    J2 = (np.sum(theta1[:,1:]**2)+np.sum(theta2[:,1:]**2))/2
    return (J1+regularization_parameter*J2)/len(y_[:,0])

def fordward_propagation(theta1,theta2,x):
    Z2 = x.dot(theta1.T)
    A2 = sigmoid(Z2)
    A2 = np.c_[np.ones(len(A2[:,0])),A2]
    Z3 = A2.dot(theta2.T)
    y_hat = sigmoid(Z3)
    return y_hat,A2

def backprop(theta1,theta2,x,y):
    y_predict,A2 = fordward_propagation(theta1,theta2,x)
    delta3 = y_predict-y
    delta2 = delta3.dot(theta2)*(A2*(1-A2))
    theta2 = theta2-learning_rate*(delta3.T.dot(A2)+
                                   regularization_parameter*np.c_[np.zeros(len(theta2[:,0])),theta2[:,1:]])/len(y[:,0])
    theta1 = theta1-learning_rate*(delta2[:,1:].T.dot(x)+
                                   regularization_parameter*np.c_[np.zeros(len(theta1[:,0])),theta1[:,1:]])/len(y[:,0])
    return theta1,theta2,y_predict

#隐藏层的可视化
def display_hidden(x,k,theta):
    im1 = Image.fromarray((x[k,1:]*255).reshape(20,20))
    im2 = Image.fromarray((x[k,1:].dot(theta[:,1:].T)*255).reshape(-1,int(np.sqrt(HIDDEN_NODE))))
    plt.subplot(211)
    plt.imshow(im1)
    plt.subplot(212)
    plt.imshow(im2)
    plt.show()


def main():
    t1 = time.perf_counter()
    x,y = read_xy()
    theta1 = np.random.randn(HIDDEN_NODE,len(x[0])) #初始化参数
    theta2 = np.random.randn(K,HIDDEN_NODE+1)
    y_ = np.zeros((len(y),K))
    for i in range(len(y)):
        for j in range(K):
            y_[i,y[i]] = 1
    for i in range(iterations):
        theta1,theta2,y_predict = backprop(theta1,theta2,x,y_)
        if i%2000 ==0:
            print('经过{}次迭代，损失函数为：{:.6f}'.format(i,cost(theta1,theta2,x,y_)))
    print('经过{}次迭代，损失函数为：{:.6f}'.format(iterations,cost(theta1,theta2,x,y_)))
    print('最终的准确率是：{:.2f}%'.format(np.sum(y_*y_predict)/len(y)*100))
    display_hidden(x,0,theta1) #可视化图片0激发的隐藏层
    t2 = time.perf_counter()
    print('程序总用时：{:.3f}s'.format(t2-t1))

#如果main是主函数，则执行main    
if __name__ == '__main__':
    main()