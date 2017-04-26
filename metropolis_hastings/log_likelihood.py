"""
filename: likelihood.py
author: Thomas Hasenzagl
date: April 24, 2016
"""

import numpy as np
import scipy.stats as stats

def log_likelihood(theta, y, x):
    """
    log likelihood function for a regression model with normally distributed errors
    """

    beta = theta[0:-1]
    sigma = theta[-1]
    n=x.shape[0]

    return -(n/2) * np.log(2*np.pi*sigma**2) - (1/(2*sigma**2))*np.dot((y-np.dot(x,beta)).T, (y-np.dot(x,beta)))

def prior(theta):
    """
    priors
    """
    alpha = theta[0]
    beta = theta[1]
    sigma = theta[2]
    
    #from IPython.core.debugger import Tracer; Tracer()()
    alpha_prior = stats.norm.logpdf(alpha, loc=0, scale=10) 
    beta_prior = stats.norm.logpdf(beta, loc=0, scale=10)
    sigma_prior = stats.norm.logpdf(sigma, loc=0, scale=30)

    return alpha_prior + beta_prior + sigma_prior

def posterior(theta, y, x):
    """
    posterior distribution
    """
    return log_likelihood(theta, y, x) + prior(theta)

def proposal(theta):
    """
    proposal distribution is normal
    """
    alpha = theta[0]
    beta = theta[1]
    sigma = theta[2]

    alpha_draw=np.random.normal(loc=alpha, scale=0.1)
    beta_draw=np.random.normal(loc=beta, scale=0.5)
    sigma_draw=np.random.normal(loc=sigma, scale=0.3)
    
    draw = np.array([alpha_draw,beta_draw,sigma_draw]) 

    return draw
