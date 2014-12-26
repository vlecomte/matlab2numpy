from time import clock
from numpy import *

from lagrange import *

# ===================
#  Utility functions
# ===================

def remove_near(x,X, eps=spacing(1)):
    x_n1, X_1m = meshgrid(x,X)
    # We only take x where it is farther than eps from all elements of X.
    return x[abs(x_n1-X_1m).min(0) > eps]

# Calls all the interpolation functions so that they are already loaded in
# memory when we benchmark them.
def call_all():
    x = array([])
    X = array([0,1])
    U = array([0,1])
    lagrange_naive(x,X,U)
    lagrange_clever(x,X,U)
    lagrange_super(x,X,U)
    lagrange_polyfit(x,X,U)
    
# Function wrapper to measure execution time. It copies function arguments
# and hands them to the given function.
def tictoc(f, *args):
    start = clock()
    f(*args)
    end = clock()
    return end-start
