#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hashemi
The following implements a simulation of the ito integral of a brownian 
function. We increase the number of steps and observe the 
convergence of the mean and variance to the theoretical values.
"""
import numpy as np
import matplotlib.pyplot as plt


def eye(x):
    return x

class brownian():
     
    ''' Each object is a brownian motion simulation where numsum is the number
    of simulations. Each row represents one specific instance. The number of 
    steps, maxtime, and timde delta is defined for the whole class. '''

    Tmax = 1

    def __init__(self, numsim, nsteps = 512):
        self.numsim = numsim            # How many simulations?
        self.nsteps = nsteps
        self.delt = brownian.Tmax/nsteps
        mat = np.triu(np.ones([nsteps,nsteps]))
        self.t = np.arange(1,nsteps+1)*self.delt
        self.steps = np.random.normal(0,np.sqrt(self.delt), \
                                      size = [numsim,nsteps])
        self.path = np.matmul(self.steps,mat)
        
    def funcB(self, func):
        funcb = np.array(list(map(func, self.path)))
        return funcb

b = brownian(10000)

def ItoInt(b, func):
    delBt = np.roll(b.steps,-1)
    delBt[:,b.nsteps-1] = 0
    funcb = b.funcB(func)
    integral = np.sum(funcb*delBt, axis = 1)
    return integral


I = ItoInt(b,eye)

print(np.mean(I), np.var(I))
mean = []
var = []
for flag in range(100,100000):
    mean.append(np.mean(I[0:flag]))
    var.append(np.var(I[0:flag], ddof = 1))

plt.plot(list(range(100,100000)),mean)
plt.plot(list(range(100,100000)),var)
