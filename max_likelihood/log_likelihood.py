"""
filename: likelihood.py
author: Thomas Hasenzagl
date: April 24, 2016
"""

import numpy as np

def log_likelihood(theta, y, x):
    """
    log likelihood function for a regression model with normally distributed errors
    """

    beta = theta[0:-1]
    sigma = theta[-1]
    n=x.shape[0]

    return -(n/2) * np.log(2*np.pi*sigma**2) - (1/(2*sigma**2))*np.dot((y-np.dot(x,beta)).T, (y-np.dot(x,beta)))
