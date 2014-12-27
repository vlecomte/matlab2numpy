from numpy import *
from matplotlib.pyplot import *

def egg(top, bottom, dt, mode=0):
    
    p = 2
    T = [0,0,0,1,1,2,2,3]
    S = [0,0,0,1,1,2,2,3,3,4,4,5]
    
    w_arc = 1 / sqrt(2)
    
    Rt = array([0,1,1,1,0])
    Ht = array([-bottom,-bottom,0,top,top])
    Wt = array([1,w_arc,1,w_arc,1])
    
    Xs = array([1, 1, 0,-1,-1,-1, 0, 1, 1])
    Ys = array([0, 1, 1, 1, 0,-1,-1,-1, 0])
    Zs = ones(shape(Xs))
    Ws = array([1,w_arc,1,w_arc,1,w_arc,1,w_arc,1])
    
    X = outer(Xs,Rt)
    Y = outer(Ys,Rt)
    Z = outer(Zs,Ht)
    W = outer(Ws,Wt)
    
    nt = len(T)-1
    t = linstep(T[p], T[nt-p], dt)
    Bt = zeros((nt-p, len(t)))
    for i in range(nt-p):
        Bt[i] = b(t,T,i,p)
    
    ns = len(S)-1
    s = linstep(S[p], S[ns-p], dt)
    Bs = zeros((ns-p, len(s)))
    for i in range(ns-p):
        Bs[i] = b(s,S,i,p)
    
    w = dotn(Bs.T, W, Bt)
    x = dotn(Bs.T, W*X, Bt) / w
    y = dotn(Bs.T, W*Y, Bt) / w
    z = dotn(Bs.T, W*Z, Bt) / w
    
    return x,y,z

def b(t,T,i,p):
    
    if p == 0:
        return (T[i] <= t) & (t < T[i+p+1])
    
    u = zeros(shape(t))
    if T[i] != T[i+p]:
        u += (t-T[i]) / (T[i+p] - T[i]) * b(t,T,i,p-1)
    if T[i+1] != T[i+p+1]:
        u += (T[i+p+1]-t) / (T[i+p+1] - T[i+1]) * b(t,T,i+1,p-1)
    
    return u

def linstep(a,b,dx):
    return linspace(a,b,(b-a)/dx)

def dotn(*args):
    p = args[0]
    for m in args[1:]:
        p = dot(p,m)
    return p
