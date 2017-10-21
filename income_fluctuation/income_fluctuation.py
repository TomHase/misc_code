import numpy as np

class income_fluctuation_model:

    def __init__(self, sigma, beta, r, T, grid_min, grid_max, h):
        self.sigma = sigma
        self.beta = beta
        self.r = r
        self.T = T
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.h = h

    def inverted_euler(self, cp):
        return (self.beta * (1 + self.r) * (1 / cp))**-1  
    
    def make_grid(self):
        return np.linspace(self.grid_min, self.grid_max, num=self.h) 

    def income_process(self): 
        return np.random.normal(scale=self.sigma, size=self.T) 

    def cash_on_hands(self):

        # Initialize arrays
        cpolicy = np.zeros([self.h, self.T])
        zgrid = np.zeros([self.h, self.T])

        # make exogenous grid for a
        agrid = self.make_grid() 
        
        # Simulate y
        y = self.income_process()

        from IPython.core.debugger import Tracer; Tracer()()

        # consume all the cash-on-hand
        cpolicy[:, self.T-1] = (1+self.r)*agrid + y[self.T-1] 
        zgrid[:,self.T-1] = cpolicy[:,self.T-1]

        for t in range(self.T-2, -1, -1):
            cpolicy[:,t] = self.inverted_euler(cpolicy[:,t+1]) 
            zgrid[:,t] = agrid + cpolicy[:,t] # endogenous grid

        return cpolicy, zgrid 

    def standard(self):

        # Initialize array
        cpolicy = np.zeros([self.h, self.T])
        astar = np.zeros([self.h, self.T])

        # make exogenous grid for a
        agrid = self.make_grid() 
        
        # Simulate y
        y = self.income_process()
         
        # consume all income
        cpolicy[:, self.T-1] = (1+self.r)*agrid + y[self.T-1] 

        #from IPython.core.debugger import Tracer; Tracer()()
        for t in range(self.T-2, -1, -1):
            cpolicy[:,t] = self.inverted_euler(cpolicy[:,t+1]) 
            astar[:,t] = (cpolicy[:,t] + agrid - y[t])/(1+self.r)    

        return cpolicy, astar





