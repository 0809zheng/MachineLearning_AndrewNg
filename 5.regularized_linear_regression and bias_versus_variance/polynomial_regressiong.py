# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 08:38:21 2019

@author: zhijiezheng
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

learning_rate = 0.1 #设定学习率
iterations = 10000 #设定梯度下降轮数
regularization_parameter = 3 #正则化系数
poly = 8 #多项式次数 

#定义函数实现文件输入
def read_xy():
    data = loadmat('ex5data1.mat')
    return data['X'],data['y'],data['Xtest'],data['ytest'],data['Xval'],data['yval']

#构造高次特征
def poly_feature(x):
    for i in range(poly-1):
        x = np.c_[x,pow(x[:,0],i+2)]
    return x

#特征缩放
def feature_scaling(x):
    return (x-np.mean(x,axis=0))/np.std(x,axis=0),np.mean(x,axis=0),np.std(x,axis=0)

#定义假设函数
def h(theta,x):
    return x.dot(theta)

#定义损失函数
def cost(theta,x,y,lambda_=regularization_parameter):
    return (np.sum((h(theta,x)-y)**2)+lambda_*np.sum(theta[1:]**2))/(2*len(y))

#实现批量梯度下降算法
def batch_gardient_descent(theta,x,y,lambda_=regularization_parameter):
    return theta-learning_rate*(x.T.dot(h(theta,x)-y)+lambda_*np.r_[[[0]],theta[1:]])/len(y)

#绘制数据和回归直线
def draw_pic(theta,x,y,miu,sigma):
    plt.scatter(x,y,marker='o',s=50,cmap='Blues',alpha=0.5) 
    plt.xlabel('水位变化',fontproperties='SimHei',fontsize=20) 
    plt.ylabel('泄洪量',fontproperties='SimHei',fontsize=20) 
    x_=np.linspace(-60,60,1000).reshape(-1,1) 
    y_=theta[0,0]
    for i in range(poly):
        y_ += theta[i+1,0]*(poly_feature(x_)[:,i]-miu[i])/sigma[i]
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
    
#绘制正则化系数曲线
def lambda_curve(train_err,cv_err):
    plt.xlabel('正则化系数',fontproperties='SimHei',fontsize=20)
    plt.ylabel('代价函数值',fontproperties='SimHei',fontsize=20)
    plt.plot(train_err,'r') 
    plt.plot(cv_err,'b') 
    plt.legend(['train','cv'])
    plt.show()    
    
#画学习曲线    
def draw_learning_curve(x,y_train,x_cv,y_cv,miu,sigma):
    m = len(x[:,0])
    train_err = np.zeros(m)
    cv_err = np.zeros(m)
    for n in range(m):
        xn = x[0:n+1,:]
        y = y_train[0:n+1].reshape(-1,1)
        theta = np.ones(len(x[0])).reshape(-1,1)
        for i in range(iterations):
            theta = batch_gardient_descent(theta,xn,y)
        train_err[n]=cost(theta,xn,y) 
        x_ = np.c_[np.ones(len(x_cv[:,0])),(poly_feature(x_cv)-miu)/sigma]
        cv_err[n]=cost(theta,x_,y_cv)
    learning_curve(train_err,cv_err)  

#画正则化系数曲线    
def draw_lambda_curve(x,y_train,x_cv,y_cv,miu,sigma):
    k = 20
    train_err = np.zeros(k)
    cv_err = np.zeros(k)
    lambda_ = np.linspace(0,10,k)
    for i in range(k):
        theta = np.ones(len(x[0])).reshape(-1,1)
        for j in range(iterations):
            theta = batch_gardient_descent(theta,x,y_train,lambda_[i])
        train_err[i] = cost(theta,x,y_train,lambda_[i])
        x_ = np.c_[np.ones(len(x_cv[:,0])),(poly_feature(x_cv)-miu)/sigma]
        cv_err[i]=cost(theta,x_,y_cv)
    lambda_curve(train_err,cv_err)
    
def main():
    x_train,y_train,x_test,y_test,x_cv,y_cv = read_xy()
    x_train_,miu,sigma = feature_scaling(poly_feature(x_train))
    x = np.c_[np.ones(len(x_train_[:,0])),x_train_]
    theta = np.ones(len(x[0])).reshape(-1,1)
    for i in range(iterations):
        theta = batch_gardient_descent(theta,x,y_train)
        if i % 2000 == 0:
            print('经过{}次迭代，损失函数为：{:.6f}'.format(i,cost(theta,x,y_train)))
    print('经过{}次迭代，损失函数为：{:.6f}'.format(iterations,cost(theta,x,y_train)))
    draw_pic(theta,x_train,y_train,miu,sigma) 
    draw_learning_curve(x,y_train,x_cv,y_cv,miu,sigma)
    draw_lambda_curve(x,y_train,x_cv,y_cv,miu,sigma)
    x_test = (poly_feature(x_test)-miu)/sigma
    x_test = np.c_[np.ones(len(x_test[:,0])),x_test]
    print('测试集误差是:{:.4f}'.format(cost(theta,x_test,y_test)))

#如果main是主函数，则执行main    
if __name__ == '__main__':
    main()