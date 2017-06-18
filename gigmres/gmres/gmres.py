"""
filename: gmres.py
author: Thomas Hasenzagl
email: thomas.hasenzagl@gmail.com
date: june 18, 2017
"""

import numpy as np
from numpy.linalg import norm

def gmres(A, b, x0, max_iter = 10):

    n = A.shape[0]
    tol = 1e-3
    k = 1

    # Initial arrays 
    q = np.zeros([n, max_iter])
    H =np. zeros([max_iter+1, max_iter])

    R0 = b - np.dot(A, x0)
    beta = norm(R0)
    error = norm(R0)

    q[:,0] = R0 / norm(R0) 

    for j in range(max_iter):
        qj = np.dot(A, q[:,j])
        
        for i in range(j):
            H[i,j] = np.dot(np.conj(qj.T), q[:,i])
            qj = qj - np.dot(H[i,j], q[:,i]) 

        if j != max_iter - 1:  
            H[j+1, j] = norm(qj)
            q[:,j+1] = qj / H[j+1,j]

        e1 = np.hstack([1, np.zeros(max_iter)])
        yk = np.linalg.lstsq(H,beta*e1)[0]
        x = np.dot(q, yk) + x0

        error = norm(np.dot(A, x) - beta)
        if error < tol:
            break

    return x 
