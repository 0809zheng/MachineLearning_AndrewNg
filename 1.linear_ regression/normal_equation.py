# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:06:37 2019

@author: zhijiezheng
"""

import time
import numpy as np

#定义函数实现文件输入
def read_xy():
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    x = np.array(data[:,:-1])
    y = np.array(data[:,-1]).reshape(-1,1) #把y变换成1列数组
    x = np.c_[np.ones(len(y)),x]
    return x,y

#正规方程法
def normal_equation(x,y):
    return np.linalg.pinv(x.T.dot(x)).dot(x.T.dot(y))

#定义损失函数
def cost(theta,x,y):
    return np.float((x.dot(theta)-y).T.dot((x.dot(theta)-y))/(2*len(y)))

def main():
    t1 = time.perf_counter()
    x,y = read_xy()
    theta = normal_equation(x,y)
    print('正规方程法得到的回归参数为：{}'.format(theta))
    print('损失函数为：{:.6f}'.format(cost(theta,x,y)))
    t2 = time.perf_counter()
    print('程序总用时：{:.3f}s'.format(t2-t1))
    print('试比较观察当样本尺寸不大时，两种方法的运算速度差距。')

#如果main是主函数，则执行main    
if __name__ == '__main__':
    main()
