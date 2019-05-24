# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:49:13 2019

@author: zhijiezheng
"""

import numpy as np
from scipy.io import loadmat
from PIL import Image
import matplotlib.pyplot as plt
import time

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

def sigmoid(x):
    return 1./(1+np.exp(-x))

def fordward_propagation(theta1,theta2,x,y):
    Z2 = x.dot(theta1.T)
    A2 = sigmoid(Z2)
    Z3 = np.c_[np.ones(len(y)),A2].dot(theta2.T)
    y_hat = sigmoid(Z3)
    return y_hat
    
def test(y_,y):
    count = 0
    #由于所给参数对应one hot编码顺序是[1，2，3，4，5，6，7，8，9，0]，故做如下调整：
    y_ =np.c_[y_[:,-1],y_[:,0:-1]]
    predict = y_//np.max(y_,axis=1).reshape(-1,1)
    for i in range(len(y)):
        if predict[i,y[i]]==1.:
            count+=1
    print('最终的准确率是：{}%'.format(count/len(y)*100))

def main():
    t1 = time.perf_counter()
    x,y = read_xy()
    display_data(x,0) #展示第0张图片
    theta = loadmat('ex3weights.mat') #加载参数
    theta1 = theta['Theta1']
    theta2 = theta['Theta2']
    y_ = fordward_propagation(theta1,theta2,x,y)
    test(y_,y)
    t2 = time.perf_counter()
    print('程序总用时：{:.3f}s'.format(t2-t1))

#如果main是主函数，则执行main    
if __name__ == '__main__':
    main()