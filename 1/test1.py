from numpy import *
from matplotlib.pyplot import *

from lagrange import *
from util import *

# ==========================
#  1. Simple execution test
# ==========================
n = 60
m = 1000
X = linspace(0,5,n+1)
# Operators on arrays are element-by-element operators by default.
U = 1 / (X+1)

# We remove sigular points (points near X).
x = remove_near(linspace(0,5,m), X)
uh = lagrange_super(x,X,U)

# The plot function works pretty much like its MATLAB equivalent. However, it
# blocks the execution until the window is closed.
plot(x,uh, X,U,'ro')
title('The super Lagrange interpolation')
show()

# ==============================
#  2. Computing efficiency test
# ==============================
print('Efficiency tests with m={}'.format(m))

# For this list and for the recorded times, no need to use numpy's arrays.
nSet = [2,5,10,20,50,100]
# Python is cool.
[na,cl,su,po] = [[0]*len(nSet) for i in range(4)]

# Will ignore warnings from polyfit. In practice it is a bad idea to use higher
# order polynomial interpolations. And possibly even worse to ignore warnings.
warnings.simplefilter('ignore', RankWarning)
# Calls all functions so that loading the function into memory will not have an
# influence on the running time.
call_all()

for i,n in enumerate(nSet):
    X = linspace(0,10,n+1)
    print('  Order of interpolation is {}'.format(n))
    
    # Again, we remove singular points.
    x = remove_near(linspace(0,10,m), X)
    U = 1 / (X+1)
    
    # Benchmarking every version. The naive version is too slow to be run above
    # degree 20.
    if n <= 20:
        na[i] = tictoc(lagrange_naive, x,X,U)
        print('    Naive lagrange:   {:.3f}'.format(na[i]))
    cl[i] = tictoc(lagrange_clever, x,X,U)
    print('    Clever lagrange:  {:.3f}'.format(cl[i]))
    su[i] = tictoc(lagrange_super, x,X,U)
    print('    Super lagrange:   {:.3f}'.format(su[i]))
    po[i] = tictoc(lagrange_polyfit, x,X,U)
    print('    Polyfit lagrange: {:.3f}'.format(po[i]))

# A log-log plot will be more readable, and complexities of the form O(C*n^k)
# will appear as lines.
loglog(nSet,na,'.-r', nSet,cl,'.-b', nSet,su,'.-g', nSet,po,'.-k')
xlabel('Order of interpolation')
ylabel('CPU time (s)')
legend(['naive','clever','super','polyfit'])
show()
