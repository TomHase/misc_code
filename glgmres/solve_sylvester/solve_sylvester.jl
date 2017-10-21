"""
filename: solve_sylvester.jl
author: Thomas Hasenzagl
email: thomas.hasenzagl@gmail.com
"""

function solve_sylvester(A::Array{Array{Float64,2},1}, B::Array{Array{Float64,2},1}, C::Array{Float64,2}, X0=nothing, tol=1e-10, maxiter=10000)
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

    n = size(C)[1] 
    p = size(C)[2]
    H = zeros(maxiter+1, maxiter+1)
    V = zeros(n, p*(maxiter+1))
    c = zeros(maxiter)
    s = zeros(maxiter)
       
    if X0 == nothing
        X0 = zeros(n,p)
    end

    R0 = C - sum([A[i]*X0*B[i] for i=1:size(A,1)]) 
    beta = vecnorm(R0) 
    epsilon = vecnorm(R0) 
    b = [beta; zeros(maxiter)]
    V[1:n, 1:p] = R0 / beta 

    for j = 1:maxiter    

        # Arnoldi algorithm 
        hj = zeros(j+1)
        Vj = sum([A[i]*V[:,(j-1)*p+1:j*p]*B[i] for i=1:size(A,1)]) 

        for i = 1:j
            hj[i] = dot(V[:,(i-1)*p+1:i*p],Vj)
            Vj -= hj[i] * V[:,(i-1)*p+1:i*p]
        end

        hj[end] = vecnorm(Vj) 
        Vj = Vj / hj[end] 
        V[:,j*p+1:(j+1)*p] = Vj 
    
        # Apply Givens rotation 
        for i=1:j-1
            temp = c[i]*hj[i] + s[i]*hj[i+1]
            hj[i+1] = -s[i]*hj[i] + c[i]*hj[i+1] 
            hj[i] = temp
        end

        # Update Givens rotation
        r = sqrt(hj[j] ^ 2 + hj[j+1] ^ 2)
        c[j] = hj[j] / r
        s[j] = c[j] * hj[j+1] / hj[j] 
        
        # Rotate hj and b with the updated Givens rotation 
        hj[j] = c[j]*hj[j] + s[j]*hj[j+1]
        hj[j+1] = 0
        H[1:j+1, j] = hj

        b[j+1] = -s[j]*b[j]
        b[j] *= c[j]

        # Compute new residual and check stopping condition
        epsilon = abs(b[j+1])
        if epsilon < tol
            y = H[1:j, 1:j]\b[1:j]
            X = V[:, 1:j*p]*kron(y, eye(p)) + X0
            return X
        end    
    end

    display("The algorithm did not converge after " * "maxiter" 
                * " iterations. The residual is " * "epsilon" * ".") 

end
