"""
filename: solve_sylvester.py
author: Thomas Hasenzagl
email: thomas.hasenzagl@gmail.com
"""

import numpy as np
from numpy.linalg import norm
from numpy.core.umath_tests import inner1d

def solve_sylvester(A, B, C, X0=None, tol=1e-10, maxiter=10000):
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

    maxiter: int, optional
         Maximum number of iterations

    Returns
    ------------------------------------
    X: (n, p) array
         Solution to the generalized sylvester equation

    References
    ------------------------------------
    Bouhamidi, A., & Jbilou, K. (2008). A note on the numerical approximate 
    solutions for generalized Sylvester matrix equations with applications. 
    Applied Mathematics and Computation, 206(2), 687-694. Chicago.
    """
    n = C.shape[0]
    p = C.shape[1]
    H = np.zeros([maxiter+1, maxiter+1])
    V = np.zeros([n, p*(maxiter+1)]) 
    c = np.zeros(maxiter)
    s = np.zeros(maxiter)
       
    if X0 == None:
        X0 = np.zeros([n,p])
    
    R0 = C - np.sum(np.dot(np.dot(Ai, X0), Bi) for Ai, Bi in zip(A, B)) 
    beta = norm(R0) 
    epsilon = norm(R0) 
    b = np.hstack([beta, np.zeros(maxiter)])
    V[0:n, 0:p] = R0 / beta 

    for j in range(1, maxiter+1):    

        # Arnoldi algorithm 
        hj = np.zeros(j+1)
        Vj = np.sum(np.dot(np.dot(Ai, V[:,(j-1)*p:j*p]), Bi) for Ai, Bi in zip(A, B))

        for i in range(j):
            hj[i] = np.sum(inner1d(V[:,i*p:(i+1)*p], Vj))
            Vj -= hj[i] * V[:, i*p:(i+1)*p]

        hj[-1] = norm(Vj) 
        Vj = Vj / hj[-1] 
        V[:,j*p:(j+1)*p] = Vj 
    
        # Apply Givens rotation 
        for i in range(j-1):
            temp = c[i]*hj[i] + s[i]*hj[i+1]
            hj[i+1] = -s[i]*hj[i] + c[i]*hj[i+1] 
            hj[i] = temp
        
        # Update Givens rotation
        r = np.sqrt(hj[j-1] ** 2 + hj[j] ** 2)
        c[j-1] = hj[j-1] / r
        s[j-1] = c[j-1] * hj[j] / hj[j-1] 
        
        # Rotate hj and b with the updated Givens rotation 
        hj[j-1] = c[j-1]*hj[j-1] + s[j-1]*hj[j]
        hj[j] = 0
        H[0:j+1, j-1] = hj

        b[j] = -s[j-1]*b[j-1]
        b[j-1] *= c[j-1]

        # Compute new residual and check stopping condition
        epsilon = np.abs(b[j])
        if epsilon < tol:
            y = np.linalg.solve(H[0:j, 0:j], b[0:j])
            X = np.dot(V[0:j*p, 0:j*p], np.kron(y, np.eye(p)).T) + X0
            return X

    print("The algorithm did not converge after " + str(maxiter) 
                + " iterations. The residual is " + str(epsilon) + ".") 


