"""
Filename: value_function_iteration.py
author: Thomas Hasenzagl, thomas.hasenzagl@gmail.com
date: 04/13/2017

Based on
https://lectures.quantecon.org/py/optgrowth.html 
and 
http://www3.nd.edu/~esims1/val_fun_iter_sp17.pdf
"""

import numpy as np
import time
from scipy.optimize import fminbound
from scipy import interpolate

def value_function_iteration(grid, tolerance, v0, alpha, beta, sigma, delta, bellman, grid_min, grid_max, A_dist, P):
    """
    Inputs:
    -----------
    grid: numpy array
        A number of grid points for the state variable 
    tolerance: float
        Stop iterations when infinity norm of two consecutive value functions is smaller than tolerance
    v0: numpy array
        initial guess for the value function
    bellman: python function
        the objective function that we are maximizing
    grid_min, grid_max: start and end value of the grid
    shocks: numpy array
        draws from the shocks

    Model Parameters:
    -----------
    alpha: output elasticity of capital
    beta: discount factor
    delta: annual capital depreciation rate
    sigma: degree of relative risk aversion
    """
    
    # initialize some variables
    inf_norm = 1 
    v=v0
    policy = np.empty_like(v)

    # count the number of iterations
    iterations = 0

    # time the function
    start = time.time()

    while inf_norm > tolerance:
        #from IPython.core.debugger import Tracer; Tracer()() 

        # initialize array for new value function
        v1=np.empty_like(v)            
        
        # intrapolate the value function
        v_fn = interpolate.interp1d(grid,v.T)       

        # loop of possible values for A
        for j, A in enumerate(A_dist):

            # loop over the grid for capital
            for i, k in enumerate(grid):
            
                # bellman equation
                bellman1 = lambda kp: bellman(kp, k, v_fn, alpha, beta, sigma, delta, A, P[j,:]) 

                # optimization with bounds kmin and kmax 
                k_star = fminbound(bellman1, grid_min, grid_max)                      
            
                # value of policy function at gridpoint i
                policy[i,j] = k_star
            
                # value of value function at gridpoind i
                v1[i,j] = - bellman1(k_star) 
            
        # check for convergence
        inf_norm = np.linalg.norm(v1-v, np.inf)        
        
        v=v1
        iterations+=1

    end=time.time()

    print("Convergence after {} iterations and {} seconds".format(iterations, round(end-start,2)))         
    return policy, v
