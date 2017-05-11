"""
Filename: arima.py
author: Thomas Hasenzagl
email: thomas.hasenzagl@gmail.com
date: May 10, 2017
"""

import numpy as np
from scipy.optimize import fmin_bfgs 

class arima:
    """
    Fits ARMA(p,q) model using maximum likelihood via Kalman filter.

    The ARMA model is given by:
        w_{t} = phi_{1} * w_{t-1} + ... + phi_{p} * w_{t-p} + e_{t} + theta_{1} * e_{t-1} + ... + theta_{q} * e_{t-q}
        e_t ~ w.n.(0, v^2)
    """

    def __init__(self, p, d, q, y):
        """
        Inputs: 
        -------------
        NAM         TYPE            DIM         DESCR
        y           numpy array     T x 1       time series vector
        p           integer                     number of autoregessive terms
        d           integer                     degree of differencing
        q           integer                     number of moving-average terms
        """
        self.p, self.d, self.q = p, d, q
        self.r = max(p,q+1)
                
        if self.d == 0:
            self. y = y
        else:
            self.y = np.diff(y, n=self.d) 

        self.t = len(self.y)
        
        # model estimation
        self.beta, self.phi, self.theta, self.L, self.epsilon, self.v2, self.alpha_last, self.P_last = self.__call__()

    def initialize(self):
        """
        Based on:
        Hannan, E. J., & Rissanen, J. (1982). Recursive estimation of mixed autoregressive-moving average order. Biometrika, 81-94. Chicago.

        Step 1:
        Estimate AR(k), with k>p+q

        Step 2:
        Estimate the ARMA model by OLS using the residuls from Step 1
        """

        # Step 1
        epsilon, order = self.initialize_step1()
        
        # Step 2
        beta = self.initialize_step2(epsilon, order)

        return beta 

    def initialize_step1(self):
       
        metric = 1e10

        for lags in range(max(1,self.p), np.int(np.ceil(np.log(len(self.y))**1.5))):
            
            # dependent variable
            y = self.y[lags:,]

            # matrix of lagged regressors
            for i in range(lags):
                if i == 0:
                    X = self.y[lags-i-1:-i-1] # maybe wrong. maybe minus lags
                else:    
                    X = np.column_stack([X, self.y[lags-i-1:-i-1]])

            # run regression for AR(lags)
            beta = np.dot(self.inverse(np.dot(X.T,X)),np.dot(X.T,y))
       
            # errors and error variance
            e = y - np.dot(X,beta)
            sigma2 = np.dot(e.T, e)/len(e) 

            # compute BIC
            bic = self.bic(sigma2, lags, self.t)

            if bic < metric:
                epsilon = e
                order = i + 1
                metric = bic
                
        return epsilon, order 
    
    def initialize_step2(self, epsilon, order):
        
        # dependent variable
        y = self.y[order+self.r:,]    

        # matrix of lagged regressors for AR(p)
        for i in range(self.p):
            if i == 0:
                X = self.y[order+self.r-i-1:-i-1,]
            else:
                X = np.column_stack([X, self.y[order+self.r-i-1:-i-1]])

        # matrix of lagged regressors for MA(q)
        for i in range(self.q):
            if i == 0 and self.p == 0:
                X = epsilon[self.r-i-1:-i-1]
            else:
                X = np.column_stack([X, epsilon[self.r-i-1:-i-1]])

        # run regression for AR(lags)
        beta = np.dot(self.inverse(np.dot(X.T,X)),np.dot(X.T,y))

        return beta 

    def state_space(self, beta):
        """
        The state space is given by
        
        measurement equation: w_{t} = Z'_{t} * alpha_{t} 
        transition equation: alpha_{t} = T * alpha_{t-1} + R * epsilon_{t}, epsilon_{t}~N(0,Q)

        where alpha_{t} is an r x 1 vector and r = max(p, q+1)
        """

        # Reshape beta
        beta = np.reshape(beta, [self.p+self.q,])

        # AR and MA coefficients
        if self.p == 0:
            phi = np.array([])
        else:    
            phi = beta[0:self.p]
        
        if self.q == 0:
            theta = np.array([])
        elif self.q == 1 and self.p == 0:
            theta = beta
        else:
            theta = beta[self.p:self.p+self.q]

        # Transition Matrix
        T1 = np.row_stack([phi[:,None], np.zeros([self.r - self.p, 1])])
        T2 = np.row_stack([np.identity(self.r-1), np.zeros([1, self.r-1])]) 
        T = np.column_stack([T1, T2])
        
        # Volatility Matrix in state equation
        R = np.row_stack([np.array([1]), theta[:,None], np.zeros([self.r-1 - self.q, 1])])
        
        # Output Matrix
        Z = np.row_stack([np.array([1]), np.zeros([self.r-1, 1])])
        
        # Covariance Matrix
        Q = np.dot(R, R.T)

        return T, R, Z, Q
    
    def log_likelihood(self, sum1, sum2):
        """
        The log-likelihood of the ARMA model is given by
        
        L = -{t * log(sum(e_t^2 / S_t)) + sum(log(S_t))} 
        """
        
        L = self.t * np.log(sum2) + sum1
        v2 = np.sqrt(sum2/self.t)

        return L, v2

    def kalman_filter(self, T, R, Z, Q, alpham, Pm, y):
        """
        Kalman filter for the estimation of the ARMA model
        """
        # Predicted state: alpha_{t|t-1} = T * alpha_{t-1|t-1}
        alphap = np.dot(T,alpham) 

        # Predicted Covariance: P_{t|t-1} = T * P_{k-1|k-1} * T' + Q
        Pp = np.dot(np.dot(T,Pm),T.T) + Q

        # Innovations: e_{t} = y_{t} - Z' * alpha_{t|t-1} 
        e = y - np.dot(Z.T,alphap)

        # Covariance matrix of y: Z' * P_{t|t-1} * Z
        sigma_yy = np.dot(np.dot(Z.T,Pp),Z)

        # Cross-Covariance matrix of x and y: P_{t|t-1} * Z 
        sigma_xy = np.dot(Pp, Z)

        # Kalman gain: K = sigma_xy * sigma_yy^{-1}
        K = sigma_xy * np.linalg.inv(sigma_yy)

        # Updates state estimate: alpha_{t|t} = alpha_{t|t-1} + K * e_{t}
        alphau = alphap + np.dot(K,e)        

        # Updates error covariance estimate: P_{t|t} = (I-K * H) * P_{t|t-1}
        n = Z.T.shape[1]
        Pu = np.dot((np.identity(n) - np.dot(K,Z.T)), Pp)        
        
        return e, sigma_yy, alphau, Pu 
    
    def objective(self, beta, save_output=False):
        """
        Objecive function to be maximized 
        """
        
        # initialize for storage
        epsilon = np.empty_like(self.y)

        # state space
        T, R, Z, Q = self.state_space(beta)

        # initial state and error covariance matrix
        alpham = np.zeros(self.r)
        Pm = np.reshape(np.dot(self.inverse(np.identity(self.r**2) - np.kron(T,T)) , np.reshape(Q,(self.r**2, 1))), (self.r, self.r)) 
        
        # initialize sum1 and sum2
        sum1 = 0
        sum2 = 0

        for t in range(self.t):
                        
            # Run Kalma filter
            e, sigma_yy, alphau, Pu = self.kalman_filter(T, R, Z, Q, alpham, Pm, self.y[t])
            
            # Sums for the Log Likelihood
            sum1 += np.log(sigma_yy)
            sum2 += e**2 / sigma_yy
           
            # epsilon
            epsilon[t] = e

            # Update alpha and P for next iteration
            alpham = alphau
            Pm = Pu

        # calculate log likelihood and error variance v^2
        L, v2 = self.log_likelihood(sum1, sum2)
        
        if save_output == False:
            return L
        elif save_output == True:
             return L, epsilon, v2, alphau, Pu    

    def __call__(self):
        """
        Maximize the objective function and get the coefficients
        """
               
        # initialize
        beta0 = self.initialize()
        
        # maximize the likelihood
        beta_star = fmin_bfgs(self.objective, beta0, disp=0)

        # final run of objective function
        L, epsilon, v2, alpha_last, P_last = self.objective(beta_star, save_output=True)

        # AR and MA coefficients
        if self.p == 0:
            phi = np.array([])
        else:    
            phi = beta_star[0:self.p]
        
        if self.q == 0:
            theta = np.array([])
        elif self.q == 1 and self.p == 0:
            theta = beta_star
        else:
            theta = beta_star[self.p:self.p+self.q]
        
        return beta_star, phi, theta, -L, epsilon, v2, alpha_last, P_last 
    
    def forecast(self,H):
        """
        Forecast ahead for periods h=1,...,H 

        alpha_{T+h|T} = T * alpha{T+h-1|T}
        P_{T+h|T} = T * P_{T+h-1|T} * T' + Q  
        """

        # State Space with 
        T, R, Z, Q = self.state_space(self.beta)

        # initializing for storage
        fcst = np.empty([H,])
        mse = np.empty([H,])
        
        # forecasts
        for h in range(H):
            if h == 0:
                alpham = self.alpha_last
                Pm = self.P_last
            else:
                alpham = alphap
                Pm=Pp
               
            # forecast
            alphap = np.dot(T,alpham)
            fcst[h] = np.dot(Z.T,alphap)

            # Mean squared error
            Pp = np.dot(np.dot(T,Pm),T.T) + Q
            mse[h] = self.v2 * Pp[0,0]
        
        return fcst, mse 

    def bic(self, sigma2=None, p=None, t=None):
        """
        Compute Bayesian information criterion for lag choice
        """
        if sigma2 == None and p == None and t == None:
            sigma2, p, t = self.sigma2, self.p, self.t

        return np.log(sigma2) + p * np.log(t)/t 

    def inverse(self,x):
        """
        Compute inverse, even for 0 dimentional arrays
        """

        if x.shape ==():
            return 1/x
        else:
            return np.linalg.inv(x)


