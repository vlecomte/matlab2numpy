from numpy import *
from matplotlib.pyplot import *
from time import clock

from lagrange import *

# ==========================
#  1. Simple execution test
# ==========================
n = 60
m = 1000
X = linspace(0,5,n+1)
# Operators on arrays are element-by-element operators by default.
U = 1 / (X+1)

# This is approximately the minimum value to avoid division by zero. The
# behaviour is not exactly well defined, so it's not just spacing(1).
eps = spacing(10)
x = linspace(eps,5-eps,m)
uh = lagrange_polyfit(x,X,U)

# The plot function works pretty much like its MATLAB equivalent. However, it
# blocks the execution until the window is closed.
plot(x,uh, X,U,'ro')
title('The super Lagrange interpolation')
show()

# ==============================
#  2. Computing efficiency test
# ==============================
print('Efficiency tests with m={}'.format(m))

nSet = [2,5,10,20,50,100]
na = zeros(shape(nSet))
cl = zeros(shape(nSet))
su = zeros(shape(nSet))
po = zeros(shape(nSet))

for i,n in enumerate(nSet):
    X = linspace(0,10,n+1)
    print('  Order of interpolation is {}'.format(n))
    
    # Function wrapper to measure execution time
    def tictoc(f, *args):
        start = clock()
        f(*args)
        end = clock()
        return end-start
    
    x = linspace(eps,10-eps,m)
    U = 1 / (X+1)
    if n <= 20:
        na[i] = tictoc(lagrange_naive, x,X,U)
        print('    Naive lagrange:   {:.3f}'.format(na[i]))
    cl[i] = tictoc(lagrange_clever, x,X,U)
    print('    Clever lagrange:  {:.3f}'.format(cl[i]))
    su[i] = tictoc(lagrange_super, x,X,U)
    print('    Super lagrange:   {:.3f}'.format(su[i]))
    po[i] = tictoc(lagrange_polyfit, x,X,U)
    print('    Polyfit lagrange: {:.3f}'.format(po[i]))

loglog(nSet,na,'.-r', nSet,cl,'.-b', nSet,su,'.-g', nSet,po,'.-k')
xlabel('Order of interpolation')
ylabel('CPU time (s)')
legend(['naive','clever','super','polyfit'])
show()
