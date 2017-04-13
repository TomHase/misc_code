"""
Filename: pca.py
author: Thomas Hasenzagl
date: April 12, 2017
"""

import numpy as np

class pca:
    """
    Computes Principal components from data
    
    Inputs
    ---------------------
    X: numpy array
    """

    def __init__(self, X):
        
        self.X=X

    def cov_matrix(self):

        """"
        Find the covariance matrix of the data
        """

        sigma = np.cov(self.X)
        return sigma

    def eigs(self):

        """
        Computes the eigenvalues and eigenvectors of the covariance matrix
        """

        sigma=self.cov_matrix()     
        D,V = np.linalg.eig(sigma)
        
        idx = D.argsort()[::-1]   
        D = D[idx]
        V = V[:,idx]
        return D,V

    def principal_comps(self):

        """
        Project the original data onto the eigenvectors
        """

        D,V= self.eigs()
        T = np.dot(V.T,self.X)                
        return T.T

    def var_explained(self):

        """
        Returns the fraction of the variance explained by the principal components
        """
        D,V = self.eigs() 
        U=D/np.sum(D)
        return U

