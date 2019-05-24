# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:06:37 2019

@author: zhijiezheng
"""

import time
import numpy as np

learning_rate = 0.01 #设定学习率
iterations = 1000 #设定梯度下降轮数

#定义函数实现文件输入
def read_xy():
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    x = np.array(data[:,:-1])
    y = np.array(data[:,-1]).reshape(-1,1) #把y变换成1列数组
    x = np.c_[np.ones(len(y)),feature_scaling(x)]
    return x,y

#特征缩放(均值标准化)
def feature_scaling(x):
    return (x-np.mean(x,axis=0))/np.std(x,axis=0)

#定义假设函数
def h(theta,x):
    return x.dot(theta) #矩阵乘法

#定义损失函数
def cost(theta,x,y):
    #向量化
    return np.float((x.dot(theta)-y).T.dot((x.dot(theta)-y))/(2*len(y)))
'''
    #非向量化
    J = 0
    for i in range(len(y)):
        J += (np.float(h(theta,x[i,:]))-y[i,0])**2/(2*len(y))
    return J
'''

#实现批量梯度下降算法
def batch_gardient_descent(theta,x,y):
    temp = theta
    for j in range(len(theta)):
        sigma = 0
        for i in range(len(y)):
            sigma += (np.float(h(theta,x[i,:])-y[i,0]))*x[i,j]
        temp[j] = theta[j]-learning_rate*sigma/len(y)
    return temp

#测试函数
def test(x):
    feature_scaling(x)
    
def main():
    t1 = time.perf_counter()
    x,y = read_xy()
    theta = np.ones(len(x[0])).reshape(-1,1) #初始化参数theta为列向量
    print(theta)
    for i in range(iterations):
        theta = batch_gardient_descent(theta,x,y)
        if i % 100 == 0:
            print('经过{}次迭代，损失函数为：{:.6f}'.format(i,cost(theta,x,y)))
    print('经过{}次迭代，损失函数为：{:.6f}'.format(iterations,cost(theta,x,y)))
    print('最终回归得到的回归参数为：{}'.format(theta))
    t2 = time.perf_counter()
    print('程序总用时：{:.3f}s'.format(t2-t1))
 

#如果main是主函数，则执行main    
if __name__ == '__main__':
    main()
