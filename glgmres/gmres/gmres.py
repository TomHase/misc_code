"""
filename: gmres.py
author: Thomas Hasenzagl
email: thomas.hasenzagl@gmail.com
date: june 18, 2017
"""

import numpy as np
from numpy.linalg import norm

def gmres(A, b, x0, max_iter = 100000):

    n = A.shape[0]
    tol = 1e-7
    
    # Use 2d numpy arrays
    A = np.array(A).reshape(A.shape[0], -1)
    b = np.array(b).reshape(b.shape[0], -1)
    x0 = np.array(x0).reshape(x0.shape[0], -1)

    # Initial arrays 
    R0 = b - np.dot(A, x0)
    beta = norm(R0)
    error = np.zeros(max_iter)
    error[0] = norm(R0)

    q = R0 / norm(R0) 

    for j in range(1, max_iter+1):
        hj = np.zeros(j+1)
        qj = np.dot(A, q[:,j-1])
        
        for i in range(j):

            # Gram-Schmidt orthogonalization 
            hj[i] = np.dot(np.conj(qj.T), q[:,i])
            qj = qj - np.dot(hj[i], q[:,i]) 

        hj[-1] = norm(qj)
        qj = qj / hj[-1] 
        q = np.hstack([q, qj.reshape(qj.shape[0], -1)])

        if j == 1:
            H = hj.reshape(hj.shape[0], -1) 
        else:
            H = np.vstack([H, np.zeros([1, H.shape[1]])])
            H = np.hstack([H, hj.reshape(hj.shape[0], -1)])
        
        # Minimization
        e1 = np.hstack([1, np.zeros(j)])
        yk = np.linalg.lstsq(H,beta*e1)[0]
        x = np.dot(q[:,:j], yk) + x0.T

        error[j] = norm(np.dot(A, x.T) - b)
        if error[j] < tol:
            break
    
    return x, error[:j+1] 



