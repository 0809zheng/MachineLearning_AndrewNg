# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:06:37 2019

@author: zhijiezheng
"""

import K_means
import numpy as np
from PIL import Image

file_name = 'data/bird_small.mat'

def main():
    x,miu,c = K_means.kmeans(file_name)
    for i in range(len(x[:,0])):
        x[i] = miu[int(c[i])]
    im = Image.fromarray(x.reshape(128,128,3))
    im.save('bird_new.png')
    
main()