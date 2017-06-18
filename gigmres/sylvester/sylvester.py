import numpy as np
from numpy.linalg import norm

def global_arnoldi(A, B, V0, k):
    # Initialize array
    # from IPython.core.debugger import Tracer; Tracer()()
    V = np.zeros([V0.shape[0], V0.shape[1], k+1])
    H = np.zeros([k+1, k])
    V0 = V0 / inner_product(V0, V0) 

    V[:,:,0] = V0
    
    for j in range(1, k):
        Vj = M(A, B, V[:,:,j-1]) 

        for i in range(j):
            Vj = Vj - inner_product(V[:,:,i], Vj) * V[:,:,i] 
            H[i, j-1] =  inner_product(V[:,:,i], Vj)             

        V[:,:,j] = Vj / inner_product(Vj, Vj) 
        H[j, j-1] = inner_product(Vj, Vj)

    return V, H 


def gigmres(A, B, C, k):
    iter = 0
    tol = 1e-8
    X0 = np.zeros([A[0].shape[0], B[0].shape[0]])
    R0 = C - M(A, B, X0)
    beta = inner_product(R0, R0)
    V1 = R0 / beta
    return R0, beta, V1

def inner_product(A, B):
    return np.sqrt(np.trace(np.dot(A.T, B)))

def M(A, B, X):
    return sum([np.dot(np.dot(Ai,X),Bi) for Ai, Bi in zip(A, B)])
