from numpy import *
import warnings

def lagrange_naive(x,X,U):
    # Not pre-allocating is not possible with numpy
    uh = zeros(shape(x))
    n = len(X)
    
    for k in range(len(x)):
        for i in range(n):
            phi = 1
            for j in range(n):
                if j != i:
                    phi *= (x[k]-X[j])/(X[i]-X[j])
            uh[k] += U[i] * phi
    
    return uh

def lagrange_clever(x,X,U):
    n = len(X)
    phi = ones((n,len(x)))
    
    for i in range(n):
        for j in range(n):
            if j != i:
                phi[i] = phi[i] * (x-X[j]) / (X[i]-X[j])
    
    return dot(U, phi)

def lagrange_super(x,X,U):
    
    X2,x2 = meshgrid(X,x)
    diff_xX = x2-X2
    prod_xX = prod(diff_xX,1)
    
    X3,X4 = meshgrid(X,X)
    diff_XX = X4 - X3 + eye(X.size)
    coeffs = U / prod(diff_XX,1)
    
    sum_prod = dot(1/diff_xX, coeffs)
    
    return prod_xX * sum_prod

def lagrange_polyfit(x,X,U):
    warnings.simplefilter('ignore', RankWarning)
    p = polyfit(X,U,len(X))
    return polyval(p,x)
