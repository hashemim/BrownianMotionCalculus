#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 21:57:18 2021
The following is a ONE STEP implementation of the pricing of a call option
@author: Hashemi
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm


T = 0.5  # time to expiration date
r = 0.01     # risk free rate
sigma = 0.25   # the volatility of the underlying asset
mu = 1 # drift
S0 = 550 # price of the underlying at time zero
K = 650 # strike price

# Now we compare with the exact solution
# d1 = (np.log(S0/K)+((r*T+0.5*(sigma**2)*T)))/(sigma*np.sqrt(T))
d1p = np.log(S0/K)+r*T+0.5*T*(sigma**2)
d1p = d1p/(sigma*np.sqrt(T))
d1 = d1p
#d2 =(np.log(S0/K)+((r-0.5*(sigma**2)*T))/(sigma*np.sqrt(T)))
d2 = d1 - sigma*np.sqrt(T)
exact = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
print(exact)
avg = []
for idx in range(100):
    s = np.random.default_rng()
    BhatT = s.normal(0,np.sqrt(T), size = (1000000,1))
    option = S0*np.exp(r*T-0.5*(sigma**2)*T)*np.exp(sigma*BhatT)-K
    option1 = np.exp(-r*T)*option*(option>0)
    value = np.average(option1)
    avg.append(value)
    estimate = np.mean(avg)
    error = 100*abs((estimate-exact)/exact)
    print(idx,'   ',np.mean(avg), '    ',error)
    
print(exact)

