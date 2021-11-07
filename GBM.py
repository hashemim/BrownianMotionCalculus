#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 21:54:29 2021

@author: Hashemi
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

def eye(x): # Test function for which we can calculate the ito integral
    return x

class randomprocess():
    def __init__(self, numsim = 10000, nsteps = 512, Tmax = 1):
        self.numsim = numsim
        self.nsteps = nsteps
        self.Tmax = Tmax
        self.deltaT = Tmax/nsteps
        self.time = np.arange(1,self.nsteps+1)*self.deltaT
        self.steps = None
        self.path = None
        
    def statistics(self):
        # This method calculates the mean and variance of the process vs. time
        average = np.mean(self.path, axis = 0)
        variance = np.var(self.path, axis = 0)
        return average, variance

    def funcB(self, func):
        funcb = np.array(list(map(func, self.path)))
        return funcb
    
    def stochasticINT(self, func):
        delBt = np.roll(self.steps,-1)
        delBt[:,self.nsteps-1] = 0
        funcb = self.funcB(func)
        integral = np.sum(funcb*delBt, axis = 1)
        return integral

class brownian(randomprocess):
    # inherits from random process class
    def __init__(self, numsim, nsteps = 512, Tmax = 1):
        super().__init__(numsim,nsteps,Tmax)
        mat = np.triu(np.ones([self.nsteps,self.nsteps]))
        self.steps = np.random.normal(0,np.sqrt(self.deltaT), \
                                      size = [self.numsim,self.nsteps])
        self.path = np.matmul(self.steps,mat)
        
        
        
class geometricBM(randomprocess):
    '''
    Inherits from the random process class, but also uses a brownian motion
    in its construction. The specific brownian process is kept with the process 
    ''' 
    def __init__(self, mu = 5, sigma = 1,numsim =10000, nsteps = 512, Tmax = 1, S0 = 550):
        super().__init__(numsim, nsteps, Tmax)
        self.Bt = brownian(numsim, nsteps, Tmax)
        self.rt = np.ones([numsim,1])*S0
        self.steps = np.zeros([numsim,nsteps])
        self.path = np.zeros([numsim,nsteps])
        for idx in range(nsteps):
            dr = sigma*self.rt*self.Bt.steps[:,idx].reshape([numsim, 1])+mu*self.rt*self.deltaT
            self.steps[:,idx] = dr.reshape(numsim)
            self.path[:,idx] = self.steps[:,idx]+self.rt.reshape(numsim)
            self.rt = self.rt+dr        
            
class geometricBM2(randomprocess):
    '''
    This is a less memory intensive realization. In each step the geometric 
    Brownian motin is crearted and propagated forward.
    ''' 
    def __init__(self, mu = 5, sigma = 1,numsim =10000, nsteps = 512, Tmax = 1, S0 = 550):
        super().__init__(numsim, nsteps, Tmax)
        Bt = brownian(numsim, nsteps, Tmax)
        self.rt = np.ones([numsim,1])*S0
        for idx in range(nsteps):
            dr = sigma*self.rt*Bt.steps[:,idx].reshape([numsim, 1])+mu*self.rt*self.deltaT
            self.rt = self.rt+dr        

# Parameters of the option. 
T = 0.5            # Expiration time (time to expiry)
S0 = 550           # Initial stock price
K = 650            # strike price
r = 0.01           # risk free rate
sigma = 0.25       # volatility
mu = 1             # drfit of the stock price (not used here)

# Parameters for the simulation
num = 100          # Number of simulation batches.
numsim = 10000     # number of instances in each batch
steps = 512       # Number of time-steps in an instance of the brownian motion


# First we add the exact solution for comparison
d1 = np.log(S0/K)+r*T+0.5*T*(sigma**2)
d1 = d1/(sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
exact = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
print(exact)    

sample = []
error = []
sum = 0
for flag in range(1,num+1):
    b = geometricBM2(0, sigma, numsim, steps,T, S0)
    STstar = b.rt
    option = STstar - math.exp(-r*T)*K
    option = option * (option>0)
    price = np.mean(option)
    sum += price
    estimate = sum/flag    
    sample.append(price)
#    estimate = np.mean(sample)
    errornext = 100*abs((estimate - exact)/exact)
    error.append(errornext)
    s = f'{flag:5}{estimate:20.12}{errornext:15.5}'
    print(s)
print(exact)

x = np.array(list(range(num)))
plt.plot(x, error)


'''
Because we do not always have access to the exact answer, below we calculate 
the variance of the sample. 
'''
std = np.std(sample, ddof = 1)
s2 = f'The confidence interval is price = {price:0.5} \u00b1 {4*std/np.sqrt(num):.5}'
print(s2)





