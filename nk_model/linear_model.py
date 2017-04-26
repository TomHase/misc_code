"""
Filename: linear_model.py

author: Thomas Hasenagl
email: thomas.hasenzagl@gmail.com
date: April 3, 2017
"""

import sympy as sp
import numpy as np
import pandas as pd

class linear_model:
    """
    Solves a linear rational expectations model and produces IRFs

    """

    def __init__(self, params, variables, system):
        """
        Model
        """

        # parameters
        self.params = params 

        # variables
        for key, value in variables.items():
            setattr(self, key, value)

        # system
        self.system = system

        
    def solve(self):
        """
        Solves the Model
        """
        
        # matrix with the system of equations
        sys = sp.Matrix(sp.sympify(self.system))

        # substitute the parameters out
        param_sym=sp.var(tuple(self.params.keys()))
        param_subs = [(param_sym[i], self.params[str(param_sym[i])]) for i in range(len(param_sym))] 
        sys=sys.subs(param_subs)

        # matrices with symbolic variables
        X = sp.Matrix(sp.var(self.var)) # at time t
        Xm = sp.Matrix(sp.var(self.varm)) # at time t-1
        Xp = sp.Matrix(sp.var(self.varp)) # at time t+1
        
        # write the system as 0 = A*X_t-1 + B*X_t + C*X_t+1
        A = sys.jacobian(Xm)
        B = sys.jacobian(X)
        C = sys.jacobian(Xp)
        A = np.array(A).astype(np.float64) # convert to numpy array
        B = np.array(B).astype(np.float64)
        C = np.array(C).astype(np.float64)
        
        # initial guess: F=0
        F=np.zeros((np.size(A,0), np.size(A,1)))
        H=np.zeros((np.size(A,0), np.size(A,1)))
        inf_norm = 1
        
        while inf_norm>1e-13:
            F = np.dot(np.linalg.inv(B + np.dot(C, F)), (-A))
            H = np.dot(np.linalg.inv(np.dot(A, H)+B), (-C))

            M = A+np.dot(B,F)+np.dot(C, np.dot(F,F))
            inf_norm = np.linalg.norm(M, np.inf)
       
        # check Blanchard and Kahn conditions
        self.check_bk(F,H)

        return F


    def check_bk(self, F, H):
        """
        Checks Blanchard Kahn conditions
        """

        bk = max(max(abs(np.linalg.eigvals(F))), abs(max(np.linalg.eigvals(H))))

        if bk > 1:
            sys.exit('Blanchard and Kahn conditions are not satisfied.')


    def policy(self):
        """
        Prints policy functions
        """

        # solve the model
        F = self.solve()

        # create pandas dataframe and print it
        df = pd.DataFrame(F, index=self.var, columns=self.varm)
        print(df)


    def irf(self, shock_var, shock_size = 1, T=9):
        """
        Computes IRFs
        """
        
        # solve the model
        F = self.solve()

        # index of variable that we shock
        idx = self.var.index(shock_var)

        # initialize array for irfs
        irf=np.zeros((np.size(F,0), 1))
        irf[idx] = shock_size
                
        irf_tm = irf
        for t in range(T-1):
            irf_t = np.dot(F, irf_tm)
            irf = np.append(irf, irf_t, axis=1)           
            irf_tm = irf_t

        return irf

