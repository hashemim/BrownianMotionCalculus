#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 13:22:24 2021
The following program simulate a random walk. 
@author: M Hashemi
"""

import random
import numpy as np
import matplotlib.pyplot as plt

n = 2**12 # number of steps
T = 1
deltaT = np.sqrt(T/n)
num_walk = 10000
mat = np.triu(np.ones([n,n]))
t = np.arange(1,n+1)/n

msteps = np.array(random.choices([-1,1],k = num_walk*n))
msteps.resize(num_walk,n)
msteps = msteps*deltaT
mpath = np.matmul(msteps,mat)

plt.figure(1)
for flag in range(10):
    plt.plot(t.reshape(n,1),mpath[flag,:].reshape(n,1), linewidth = 0.5, linestyle = 'solid')

plt.figure(2)
n, bins, patches = plt.hist(mpath[:,n-1], 1000, histtype = 'step', cumulative = False)