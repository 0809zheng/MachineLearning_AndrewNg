# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:21:25 2019

@author: zhijiezheng
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

learning_rate = 0.0005 #设定学习率
iterations = 10000 #设定梯度下降轮数
regularization_parameter = 0 #正则化系数

#定义函数实现文件输入
def read_xy():
    data = loadmat('ex5data1.mat')
    return data['X'],data['y'],data['Xtest'],data['ytest'],data['Xval'],data['yval']

def h(theta,x):
    return x.dot(theta)

#定义损失函数
def cost(theta,x,y):
    return (np.sum((h(theta,x)-y)**2)+regularization_parameter*np.sum(theta[1:]**2))/(2*len(y))

#实现批量梯度下降算法
def batch_gardient_descent(theta,x,y):
    return theta-learning_rate*(x.T.dot(h(theta,x)-y)+regularization_parameter*np.r_[[[0]],theta[1:]])/len(y)

#绘制数据和回归直线
def draw_pic(theta,x,y):
    plt.scatter(x,y,marker='o',s=50,cmap='Blues',alpha=0.5) 
    plt.xlabel('水位变化',fontproperties='SimHei',fontsize=20) 
    plt.ylabel('泄洪量',fontproperties='SimHei',fontsize=20)
    x_=np.linspace(-50,50,1000)
    y_=theta[0,0]+theta[1,0]*x_ 
    plt.plot(x_,y_,'r')  
    plt.show()

#绘制学习曲线
def learning_curve(train_err,cv_err):
    plt.xlabel('训练样本数',fontproperties='SimHei',fontsize=20) 
    plt.ylabel('代价函数值',fontproperties='SimHei',fontsize=20) 
    plt.plot(train_err,'r')   
    plt.plot(cv_err,'b') 
    plt.legend(['train','cv'])
    plt.show()
    
#画学习曲线    
def draw_learning_curve(x,y_train,x_cv,y_cv):
    m = len(x[:,0])
    train_err = np.zeros(m)
    cv_err = np.zeros(m)
    for n in range(m):
        xn = x[0:n+1,:]
        yn = y_train[0:n+1].reshape(-1,1)
        theta = np.ones(len(x[0])).reshape(-1,1)
        for i in range(iterations):
            theta = batch_gardient_descent(theta,xn,yn)
        train_err[n]=cost(theta,xn,yn) 
        x_ = np.c_[np.ones(len(x_cv[:,0])),x_cv]
        cv_err[n]=cost(theta,x_,y_cv)
    learning_curve(train_err,cv_err)
   
    
def main():
    x_train,y_train,x_test,y_test,x_cv,y_cv = read_xy()
    x = np.c_[np.ones(len(x_train[:,0])),x_train]
    theta = np.ones(len(x[0])).reshape(-1,1)
    for i in range(iterations):
        theta = batch_gardient_descent(theta,x,y_train)
        if i % 2000 == 0:
            print('经过{}次迭代，损失函数为：{:.6f}'.format(i,cost(theta,x,y_train)))
    print('经过{}次迭代，损失函数为：{:.6f}'.format(iterations,cost(theta,x,y_train)))
    draw_pic(theta,x_train,y_train) 
    draw_learning_curve(x,y_train,x_cv,y_cv)

#如果main是主函数，则执行main    
if __name__ == '__main__':
    main()