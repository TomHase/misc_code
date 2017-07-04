"""
filename: sylvester.py
author: Thomas Hasenzagl
email: thomas.hasenzagl@gmail.com
date: june 18, 2017
"""

import numpy as np
from numpy.linalg import norm

def gigmres(A, B, C, X0 = [], k  = 10, tol = 1e-7):
    
    # Dimensions of the arrays
    n = C.shape[0]
    p = C.shape[1]
        
    # Set X0  
    if X0 == []:
        X0 = np.zeros([n,p])

    # Set iter
    iter = 0 

    # Compute R0 
    R0 = C - sum([np.dot(np.dot(Ai, X0), Bi) for Ai, Bi in zip(A, B)]) 

    # Compute beta
    beta = norm(R0) 
    epsilon = beta
    epsilon_array = epsilon

    # Compute V[0]
    V = [[] for i in range(k+1)]
    V[0] = R0 / beta 

    while epsilon > tol:

        for j in range(1, k+1):
            hj = np.zeros(j+1)
            Vj = sum([np.dot(np.dot(Ai, V[j-1]), Bi) for Ai, Bi in zip(A, B)]) 
        
            for i in range(j):

                # Gram-Schmidt orthogonalization 
                hj[i] = np.trace(np.dot(V[i].T, Vj))
                Vj = Vj - np.dot(hj[i], V[i]) 

            hj[-1] = norm(Vj) 
            Vj = Vj / hj[-1] 
            V[j] = Vj 

            if j == 1:
                H = hj.reshape(hj.shape[0], -1) 
            else:
                H = np.vstack([H, np.zeros([1, H.shape[1]])])
                H = np.hstack([H, hj.reshape(hj.shape[0], -1)])

        #from IPython.core.debugger import Tracer; Tracer()()

        # Minimization
        e1 = np.hstack([1, np.zeros(k)])
        y = np.linalg.lstsq(H,beta*e1)[0]
        X = np.dot(np.hstack(V[0:-1]), np.kron(y, np.eye(p)).T) + X0

        # Error
        R = C - sum([np.dot(np.dot(Ai, X), Bi) for Ai, Bi in zip(A, B)]) 
        epsilon = norm(R)
        epsilon_array = np.vstack([epsilon_array, epsilon])

        X0 = X
        R0 = R
        beta = norm(R0)
        V[0] = R0 / beta
        iter += 1
        print(epsilon)

    return X, iter, epsilon_array 



