from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D

from egg import egg

x,y,z = egg(2,1,0.1)
ax = figure().gca(projection='3d')
ax.plot_surface(x,y,z, rstride=1,cstride=1)
ax.set_aspect('equal')
ax.set_xlim(-1.5,1.5)
ax.set_ylim(-1.5,1.5)
axis('off')
show()
