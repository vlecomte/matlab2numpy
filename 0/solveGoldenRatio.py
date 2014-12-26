from numpy import *
import scipy.linalg

def solveGoldenRatio():
    phi = (1+sqrt(5))/2
    U = array([1,1])
    A = array([[1,1],[phi,1-phi]])
    return linalg.solve(A,U)
