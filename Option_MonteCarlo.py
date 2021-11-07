#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Nov 2021
The following is a code to price an european call option. We use a monte-Carlo
approach and compare the results to the exact answer to the solution of the 
Back-Scholes equation.

One can use the exact solution for calibration of the number of time-steps for 
each realiziation of the browninan motion. 

To create the brownian motion we define two classes: 
    1. random process class (randomprocess)
    2. Brownian motion class which inherits form randomprocess
@author: Mohammad Hashemi
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
steps = 4096       # Number of time-steps in an instance of the brownian motion

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
    b = brownian(numsim,steps, T)
    BhT = b.path[:,steps-1]
    STstar = S0*np.exp(-0.5*T*(sigma**2))*np.exp(sigma*BhT)
    option = STstar - math.exp(-r*T)*K
    option = option * (option>0)
    price = np.mean(option)
    sum += price
    estimate = sum/flag    
#    sample.append(price)
#    estimate = np.mean(sample)
    errornext = 100*abs((estimate - exact)/exact)
    error.append(errornext)
    s = f'{flag:5}{estimate:20.12}{errornext:15.5}'
    print(s)
print(exact)

x = np.array(list(range(num)))
plt.plot(x, error)






