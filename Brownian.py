#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:14:31 2021
@author: M Hashemi
This program needs NUMPY and MATPLOTLIB libraries for correct execution.
"""

import numpy as np
import matplotlib.pyplot as plt

class brownian():
     
    ''' Each object is a brownian motion simulation where numsum is the number
    of simulations. Each row represents one specific instance. The number of 
    steps, maxtime, and timde delta is defined for the whole class. '''

    nsteps = 1000
    Tmax = 1
    delt = Tmax/nsteps
    mat = np.triu(np.ones([nsteps,nsteps]))
    t = np.arange(1,nsteps+1)*delt

    def __init__(self, numsim):
        self.numsim = numsim            # How many simulations?
        self.steps = np.random.normal(0,np.sqrt(brownian.delt), \
                                      size = [numsim,brownian.nsteps])
        self.path = np.matmul(self.steps,brownian.mat)


#%% Simulation
b = brownian(10000)
n = brownian.nsteps
t = brownian.t

#%% Plotting the simulations
fig, axes = plt.subplots(4) # We plot five examples of the process
for flag in range(5):
    axes[0].plot(t.reshape(n,1),b.path[flag,:].reshape(n,1),\
             linewidth = 1, linestyle = 'solid')        
    axes[0].set_ylim([-2,2]) 

'''Now we calculate and plot the MEAN and VARIANCE of the simulations. Notice 
 that the mean remains zero and the temporal increase of the variance, as
 expected '''

avg = np.mean(b.path, axis = 0)
var = np.var(b.path,axis = 0)
axes[1].plot(t.reshape(n,1),avg.reshape(n,1),linewidth = 1, linestyle = 'solid')
axes[1].plot(t.reshape(n,1),var.reshape(n,1),linewidth = 1, linestyle = 'solid')

''' Then we plot histogram of the final position on the path B(Tmax) and its 
    cumulative histogram '''
axes[2].hist(b.path[:,n-1], 512, histtype = 'bar', cumulative = False)
axes[3].hist(b.path[:,n-1], 512, histtype = 'step', cumulative = True)

"""
In the above code one can use the path of the brownian motion to create any 
function on that path and create processes that are functions of the brownian
motion and find their expectation values 
"""