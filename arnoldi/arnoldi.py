import numpy as np
from numpy.linalg import norm

def anoldi(A, q1):
    # Initialize array
    #from IPython.core.debugger import Tracer; Tracer()()
    k=len(q1)
    q = np.zeros([len(q1),k])
    q1 = q1 / norm(q1)
    q[:,0] = q1
    
    for j in range(1, k):
        qj = np.dot(A, q[:,j-1])
        
        for i in range(j):
            qj = qj - np.dot(np.conj(qj.T), q[:,i]) * q[:,i]
        
        q[:,j] = qj / norm(qj) 

    H = np.dot(np.dot(np.conj(q.T), A), q) 
    return q, H
