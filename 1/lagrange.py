from numpy import *
import warnings

# ========================
#  Lagrange interpolation
# ========================

def lagrange_naive(x,X,U):
    # Not pre-allocating is not possible with numpy, as far as I know.
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
                # Assigning the i-th line of phi.
                phi[i] = phi[i] * (x-X[j]) / (X[i]-X[j])
    
    # Matrix multiplication is done with dot(), not with '*'.
    return dot(U, phi)

def lagrange_super(x,X,U):
    # X_m1 means X replicated m times in dimension 0 and 1 time in dimension 1,
    # and so forth.
    X_m1, x_1n = meshgrid(X,x)
    diff_xX = x_1n - X_m1
    prod_xX = prod(diff_xX,1)
    
    X_n1, X_1n = meshgrid(X,X)
    # We replace the zeros on the diagonal with ones so we can take the product
    # of the lines directly.
    diff_XX = X_1n - X_n1 + eye(len(X))
    coeffs = U / prod(diff_XX,1)
    
    sum_prod = dot(1/diff_xX, coeffs)
    
    return prod_xX * sum_prod

def lagrange_polyfit(x,X,U):
    # Those functions are used the same way as in MATLAB.
    p = polyfit(X,U,len(X))
    return polyval(p,x)

# ===================
#  Utility functions
# ===================

def remove_near(x,X, eps=spacing(1)):
    x_n1, X_1m = meshgrid(x,X)
    # We only take x where it is farther than eps from all elements of X.
    return x[abs(x_n1-X_1m).min(0) > eps]

def call_all():
    # Calling all the functions so that they are already loaded in memory when
    # we benchmark them.
    x = array([])
    X = array([0,1])
    U = array([0,1])
    lagrange_naive(x,X,U)
    lagrange_clever(x,X,U)
    lagrange_super(x,X,U)
    lagrange_polyfit(x,X,U)
