# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:06:37 2019

@author: zhijiezheng
"""

import time
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.01 #设定学习率
iterations = 10000 #设定梯度下降轮数

#定义函数实现文件输入
def read_xy():
    data = np.loadtxt('ex2data1.txt', delimiter=',')
    x = np.array(data[:,:-1])
    y = np.array(data[:,-1]).reshape(-1,1) #把y变换成1列数组
    x = np.c_[np.ones(len(y)),feature_scaling(x)]
    return x,y

#特征缩放
def feature_scaling(x):
    return (x-np.mean(x,axis=0))/np.std(x,axis=0)

#定义假设函数
def h(theta,x):
    return x.dot(theta)

def sigmoid(theta,x):
    return 1./(1+np.exp(-h(theta,x)))

#定义损失函数
def cost(theta,x,y):
    J = 0
    for i in range(len(y)):
        cost1 = y[i,0]*np.log(np.float(sigmoid(theta,x[i,:])))
        cost2 = (1-y[i,0])*np.log(1-np.float(sigmoid(theta,x[i,:])))
        J += -(cost1+cost2)/(len(y))
    return np.float(J)

#实现批量梯度下降算法
def batch_gardient_descent(theta,x,y):
    temp = theta
    for j in range(len(theta)):
        sigma = 0
        for i in range(len(y)):
            sigma += (np.float(sigmoid(theta,x[i]))-y[i,0])*x[i,j]
        temp[j] = theta[j]-learning_rate*sigma/len(y)
    return temp

#绘制数据和决策边界
def draw_pic(theta,x,y,y_c):
    plt.scatter(x[:,1],x[:,2],marker='o',s=50,c=np.squeeze(y_c),alpha=0.3) #绘制输入数据的散点图 
    plt.xlabel('X1')  #设置x轴标题
    plt.ylabel('X2') #设置y轴标题
    
    x1 = np.linspace(-2,2,100)
    x2 = np.linspace(-2,2,100)
    X1,X2 = np.meshgrid(x1,x2)
    plt.contour(X1,X2,theta[0,0]+theta[1,0]*X1+theta[2,0]*X2,levels=[0]) #绘制决策边界
    plt.show()   
    
    
def main():
    t1 = time.perf_counter()
    x,y = read_xy()
    theta = np.ones(len(x[0])).reshape(-1,1) #初始化参数
    for i in range(iterations):
        theta = batch_gardient_descent(theta,x,y)
        if i % 1000 == 0:
            print('经过{}次迭代，损失函数为：{:.6f}'.format(i,cost(theta,x,y)))
    print('经过{}次迭代，损失函数为：{:.6f}'.format(iterations,cost(theta,x,y)))
    print('最终回归得到的回归参数为：{}'.format(theta))
    t2 = time.perf_counter()
    print('程序总用时：{:.3f}s'.format(t2-t1))
    y_c = [['red' if y else 'blue'] for y in y] #正样本为红色，负样本为蓝色
    draw_pic(theta,x,y,y_c)

#如果main是主函数，则执行main    
if __name__ == '__main__':
    main()