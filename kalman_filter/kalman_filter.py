"""
filename: kalman_filter.py
author: Thomas Hasenzagl
email: thomas.hasenzagl@gmai;.com
date: April, 26 2016
"""

import numpy as np

class kalman_filter:
    """
    Implements the Kalman Filter with Gaussian noise

    The state space is given by

    y_{t} = H * x_{t} + v_{t}, where v_{t} ~ N(0, R_{t}) 
    x_{t} = F * x_{t-1} + w_{t}, where w_{t} ~ N(0, Q_{t})
    x_0 ~ N(mu_0,Sigma_0)
    """ 
    def __init__(self, H, F, Q, R, x0, P0):
        """
        ________________________________________________________________
        Inputs:

        NAME    TYPE                 DESCR   
        H       numpy array         Observation Matrix
        F       numpy array         Transition Matrix 
        Q       numpy array         Covariance of error in transition equation
        R       numpy array         Covariance of error in observation equation
        x0      numpy array         initial state 
        P0      numpy array         initial error covariance 
        _______________________________________________________________

        """
        
        self.H, self.F, self.Q, self.R, self.x0, self.P0  = H, F, Q, R, x0, P0 

    def predict(self, xm, Pm):
        """ 
        Predict the state at time t using the estimate of the state at time t-1. 
        Predict the error covariance of the state at time t using the error covariance at time t-1.
        """

        # Predicted state: x_{t|t-1} = F * x_{t-1|t-1}
        xp = np.dot(self.F,xm) 

        # Predicted Covariance: P_{t|t-1} = F * P_{k-1|k-1} * F' + Q
        Pp = np.dot(np.dot(self.F,Pm),self.F.T) + self.Q
        
        return xp, Pp

    def update(self, xp, Pp, y):
        """
        Update the state and error covariance incorporating the observation y_{t} 
        """
        
        # Innovations: e_{t} = y_{t} - H * x_{t|t-1} 
        e = y - np.dot(self.H,xp)

        # Covariance matrix of y: H * P_{t|t-1} H' + R
        sigma_yy = np.dot(np.dot(self.H,Pp),self.H.T) + self.R

        # Cross-Covariance matrix of x and y: P_{t|t-1} * H'
        sigma_xy = np.dot(Pp, self.H.T)

        # Kalman gain: K = sigma_xy * sigma_yy^{-1}
        K = sigma_xy * np.linalg.inv(sigma_yy)

        # Updates state estimate: x_{t|t} = x_{t|t-1} + K * e_{t}
        xu = xp + np.dot(K,e)        

        # Updates error covariance estimate: P_{t|t} = (I-K * H) * P_{t|t-1}
        n = self.H.shape[0]
        Pu = np.dot((np.identity(n) - np.dot(K,self.H)), Pp)        
        
        return xu, Pu

    def run(self, y):
        """
        Runs the kalman filter once for every new observation in y 
        """
        xm = self.x0
        Pm = self.P0
        
        n = xm.shape[0]
        T = y.shape[0]

        xf = np.empty([T,n]) 
        Pf = np.empty([T,n,n]) 
        # from IPython.core.debugger import Tracer; Tracer()()
      
        i=0        
        for row in y:

            # Reshape and take transpose
            row = np.reshape(row, (n,1))

            # find x_{t|t-1} and P_{t|t-1}
            xp, Pp = self.predict(xm, Pm)

            # find x_{t|t} and P_{t|t}
            xu, Pu = self.update(xm, Pm, row)  
            
            # store results
            xf[i,:] = xu.T
            Pf[i,:,:] = Pu
            
            xm = xu
            Pm = Pu
            i +=1

        return xf, Pf            
