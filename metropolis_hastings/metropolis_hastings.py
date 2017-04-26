"""
Filename: metropolis_hastings.py

autor: Thomas Hasenzagl
email: thomas.hasenzagl@gmail.com
date: April 18, 2017
"""

import numpy as np

def metropolis_hastings(init, ndraws, burnin, proposal, target, y, x):
    """
    Runs the Metropolis Hastings algorithm to sample from a probablity distribution

    Inputs:
    ----------------
    initial: numpy array
        initial state
    n_draws: float
        total number of draws
    burnin: float
        number of burnin draws
    proposal: function
        pdf of the proposal density g(x)
    target: function
        log-pdf of the target density P(x)
    """

    # Initialization
    chain = np.empty([ndraws+1,init.shape[0]])
    chain[0,:]=init
    P = target(chain[0,:], y, x)
    acceptance=0

    for i in range(1,ndraws+1):

        # draw x' from proposal g(x'|x)
        draw = proposal(chain[i-1,:])

        # Target Density
        Pp = target(draw, y, x) #P(x')
        
        # acceptance distribution A(x'|x)=min(1, P(x')/P(x) * (g(x|x')/g(x'|x))
        A = np.minimum(1, np.exp(Pp - P)) 

        # accept or reject
        if np.random.uniform()<A:
            chain[i,:]=draw
            if i>burnin:
                acceptance+=1
        else:
            chain[i,:]=chain[i-1,:]
        
        if i%1000 == 0:
            print("Iteration {} with posterior likelihood {}".format(i,target(chain[i,:], y, x)))

        P=Pp

    return chain[burnin:], acceptance/(ndraws-burnin)
