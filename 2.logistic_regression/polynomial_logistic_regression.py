# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:06:37 2019

@author: zhijiezheng
"""

import time
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.1 #设定学习率
iterations = 5000 #设定梯度下降轮数
regularization_parameter = 0.01 #正则化系数

#定义函数实现文件输入
def read_xy():
    data = np.loadtxt('ex2data2.txt', delimiter=',')
    x = np.array(data[:,:-1])
    y = np.array(data[:,-1]).reshape(-1,1) #把y变换成1列数组
    x = np.c_[np.ones(len(y)),feature_map(feature_scaling(x))] 
    return x,y

def feature_map(x):
    #二次多项式回归的参数表[x0=1,x1,x2,x1^2,x2^2,x1*x2]
    return np.c_[x[:,0],x[:,1],x[:,0]*x[:,0],x[:,1]*x[:,1],x[:,0]*x[:,1]]

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
    J1,J2 = 0,0
    for i in range(len(y)):
        cost1 = y[i,0]*np.log(np.float(sigmoid(theta,x[i,:])))
        cost2 = (1-y[i,0])*np.log(1-np.float(sigmoid(theta,x[i,:])))
        J1 += -(cost1+cost2)/(len(y))
    for j in theta[1:]:
        J2 += regularization_parameter*pow(j,2)/(2*len(y))
    J = np.float(J1+J2)
    return J

#实现批量梯度下降算法
def batch_gardient_descent(theta,x,y):
    temp = theta
    for j in range(len(theta)):
        sigma = 0
        for i in range(len(y)):
            sigma += (np.float(sigmoid(theta,x[i]))-y[i,0])*x[i,j]
        delay = (1-learning_rate*regularization_parameter/len(y)) if j else 1
        temp[j] = theta[j]*delay-learning_rate*sigma/len(y)
    theta = temp
    return theta

#绘制数据和决策边界
def draw_pic(theta,x,y,y_c):
    plt.scatter(x[:,1],x[:,2],marker='o',s=50,c=np.squeeze(y_c),alpha=0.3) #绘制输入数据的散点图 
    plt.xlabel('X1')  #设置x轴标题
    plt.ylabel('X2') #设置y轴标题
    
    x1 = np.linspace(-2,2,100)
    x2 = np.linspace(-2,2,100)
    X,Y = np.meshgrid(x1,x2)
    plt.contour(X,Y,theta[0,0]+theta[1,0]*X+theta[2,0]*Y+
                theta[3,0]*pow(X,2)+theta[4,0]*pow(Y,2)+theta[5,0]*X*Y,levels=[0]) #绘制决策边界
    plt.show()   
    
    
def main():
    t1 = time.perf_counter()
    x,y = read_xy()
    theta = np.ones(len(x[0])).reshape([-1,1]) #初始化参数
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