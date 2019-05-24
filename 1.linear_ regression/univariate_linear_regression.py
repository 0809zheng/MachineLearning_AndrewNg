# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:06:37 2019

@author: zhijiezheng
"""

import time
#time库用于计时

import numpy as np
import matplotlib.pyplot as plt
#np和plt库用于绘图

learning_rate = 0.01 #设定学习率
iterations = 10000 #设定梯度下降轮数

'''
#定义函数实现不定长坐标点输入
def input_xy():
    x = []
    y = []
    input_str =input('请输入x和y，用英文逗号隔开：')
    while input_str != '':
        x.append(eval(input_str.split(',')[0]))
        y.append(eval(input_str.split(',')[1]))
        input_str =input('请输入x和y，用英文逗号隔开：')
    return x,y
'''

#定义函数实现文件输入
def read_xy():
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    return x,y #x,y均为一维数组

#定义假设函数
def h(theta0,theta1,x):
    return theta0+theta1*x

#定义损失函数
def loss(theta0,theta1,x,y):
    J = 0
    for i in range(len(x)):
        J += (h(theta0,theta1,x[i])-y[i])**2/(2*len(x))
    return J

#实现批量梯度下降算法
def batch_gardient_descent(theta0,theta1,x,y):
    sum0,sum1 = 0.,0.
    for i in range(len(x)):
        sum0 += h(theta0,theta1,x[i])-y[i]
    for j in range(len(x)):
        sum1 += (h(theta0,theta1,x[j])-y[j])*x[j]
    theta0 = theta0-learning_rate*sum0/len(x)
    theta1 = theta1-learning_rate*sum1/len(x)
    return theta0,theta1
    
#绘制数据和回归直线
def draw_pic(theta0,theta1,x,y):
    plt.scatter(x,y,marker='o',s=50,cmap='Blues',alpha=0.5) #绘制输入数据的散点图 
    plt.xlabel('X轴',fontproperties='SimHei',fontsize=20)  #设置x轴标题
    plt.ylabel('Y轴',fontproperties='SimHei',fontsize=20) #设置y轴标题
    x_=np.linspace(3,25,10000) #设置自变量取值
    y_=h(theta0,theta1,x_) #表示因变量y
    plt.plot(x_,y_,'r') #绘制回归直线    
    plt.show()
    
def main():
    t1 = time.perf_counter()
    x,y = read_xy()
    theta0,theta1 = 1.,1. #初始化参数
    for i in range(iterations):
        theta0,theta1 = batch_gardient_descent(theta0,theta1,x,y)
        if i % 1000 == 0:
            print('经过{}次迭代，损失函数为：{:.13f}'.format(i,loss(theta0,theta1,x,y)))
    print('最终回归得到的函数为：h={:.6f}+{:.6f}x'.format(theta0,theta1))
    t2 = time.perf_counter()
    print('程序总用时：{:.3f}s'.format(t2-t1))
    draw_pic(theta0,theta1,x,y) 

#如果main是主函数，则执行main    
if __name__ == '__main__':
    main()