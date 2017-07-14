"""
filename: generalized_sylvester.py
author: Thomas Hasenzagl
email: thomas.hasenzagl@gmail.com
"""

import numpy as np
from numpy.linalg import norm

def generalized_sylvester(A, B, C, X0 = [], tol = 1e-10, restart = [], maxiter=10000):
    """
    Computes the solution (X) to the generalized sylvester equation: 
    
    \sum_{i=1}^{q} A_{i} X B_{i} = C 

    Parameters
    ------------------------------------
    A: list of q (n, n) arrays 
         Leading matrices of the generalized sylvester equation 

    B: list of q (p, p) arrays
         Trailing matrices of the generalized sylvester equation

    C: (n, p) array
         Right hand side matrix

    X0: (n, p) array, optional
         Initial guess

    tol: float, optional 
         Error tolerance

    restart: int, optional
         Restart the algorithm every restart inner iterations

    maxiter: int, optional
         Maximum number of iterations

    Returns
    ------------------------------------
    X: (n, p) array
         Solution to the generalized sylvester equation

    epsilon_array: (number of iterations, 1) array
         Array of the errors

    References
    ------------------------------------
    Bouhamidi, A., & Jbilou, K. (2008). A note on the numerical approximate solutions for generalized Sylvester matrix equations with applications. Applied Mathematics and Computation, 206(2), 687-694. Chicago.

    """
    
    n = C.shape[0]
    p = C.shape[1]
        
    if X0 == []:
        X0 = np.zeros([n,p])

    R0 = C - sum([np.dot(np.dot(Ai, X0), Bi) for Ai, Bi in zip(A, B)]) 
    beta = epsilon = epsilon_array = norm(R0) 
    V = [R0 / beta] 
    
    j = 0
    k = 0
    while epsilon > tol:
        
        j += 1
        k += 1

        hj = np.zeros(j+1)
        Vj = sum([np.dot(np.dot(Ai, V[j-1]), Bi) for Ai, Bi in zip(A, B)]) 

        # Arnoldi Algorithm 
        for i in range(j):
           hj[i] = np.trace(np.dot(V[i].T, Vj))
           Vj = Vj - np.dot(hj[i], V[i]) 

        hj[-1] = norm(Vj) 
        Vj = Vj / hj[-1] 
        V.append(Vj) 

        if j == 1:
            H = hj.reshape(hj.shape[0], -1) 
        else:
            H = np.vstack([H, np.zeros([1, H.shape[1]])])
            H = np.hstack([H, hj.reshape(hj.shape[0], -1)])

        # Minimization
        e1 = np.hstack([1, np.zeros(j)])
        y = np.linalg.lstsq(H,beta*e1)[0]
        X = np.dot(np.hstack(V[0:-1]), np.kron(y, np.eye(p)).T) + X0

        # Compute Error
        R = C - sum([np.dot(np.dot(Ai, X), Bi) for Ai, Bi in zip(A, B)]) 
        epsilon = norm(R)
        epsilon_array = np.vstack([epsilon_array, epsilon])
        
        if k >= maxiter:
            print("The algorithm did not converge after " + str(maxiter) + " iterations. The residual is " + str(epsilon) + ".") 
            break

        # Restart
        if restart != []:
            if j == restart:
                j = 0
                X0 = X
                R0 = R 
                beta = norm(R0) 
                V = [R0 / beta] 

    return X, epsilon_array 

