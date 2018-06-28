


import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d




def dx (z=0.0, y=0.0):
    
    
    return -y * -z

def dy (x=0.0, a=0.0, y=0.0):
    
    
    return x + (a*y)


def dz (x=0.0, b=0.0, c=0.0, z=0.0):
    
    return b + (z*(x-c))



# might have not been the best idea
def rossler(a, b, c, x, y, z):
    
    return (np.array(dx(z, y)), np.array(dy(x, a, y)), np.array((dz(x, b, c, z))))
    
    

def rk4test(func, h=0.1, yCurrent=0.1, *funcArgs):
    #h=0.1
    """
    Function performs one RK4 step returns the value for the next iteration.
    
    Keywords:
    func: right hand side of the differential equation. 
          Default is the radioactivity function i.e. radioactive().
    h: Input. This is the time step used to the advance the solution to the next time point.
    yCurrent: Value of variable at current time step.
    *funcArgs: Parameters for the function in the differential equation.
        
    yNext: Output. Value of the variable at the next time step.
    
    """
    
    # Initialize
    yNext = 0.0
    
    # Calculate k1, k2, k3 and k4
    
    k1 = func(yCurrent, *funcArgs)
    k2 = func(yCurrent + h*k1/2, *funcArgs)
    k3 = func(yCurrent + h*k2/2, *funcArgs)
    k4 = func(yCurrent + h*k3, *funcArgs)
    
    # Calculate next time step
    yNext = (h/6)*(k1 + 2*k2 + 2*k3 + k4) + yCurrent
    
    return yNext


numIter = 50000

xss = np.zeros((numIter,1))
yss = np.zeros((numIter,1))
zss = np.zeros((numIter,1))

xTesting = np.zeros((numIter,1))
yTesting = np.zeros((numIter,1))
zTesting = np.zeros((numIter,1))

h = 0.01

def num_rossler(x_n,y_n,z_n,h,a,b,c):
    x_n1=x_n+h*(-y_n-z_n)
    y_n1=y_n+h*(x_n+a*y_n)
    z_n1=z_n+h*(b+z_n*(x_n-c))   
    return x_n1,y_n1,z_n1



def rosslerRK4(a, b, c, h, x, y, z):
    
    
    # Calculate k1, k2, k3 and k4
    
    xk1 = dx(z, y)
    xk2 = dx(z + h*xk1/2, y + h*xk1/2)
    xk3 = dx(z + h*xk2/2, y + h*xk2/2)
    xk4 = dx(z + h*xk3, y + h*xk3)
    
    
    yk1 = dy(x, a, y)
    yk2 = dy(x + h*yk1/2, a, y + h*yk1/2)
    yk3 = dy(x + h*yk2/2, a, y + h*yk2/2)
    yk4 = dy(x + h*yk3, a, y + h*yk3)
    
    
    zk1 = dz(x, b, c, z)
    zk2 = dz(x + h*zk1/2, b, c, z + h*zk1/2)
    zk3 = dz(x + h*zk2/2, b, c, z + h*zk2/2)
    zk4 = dz(x + h*zk3, b, c, z + h*zk3)
    
    # Calculate next time step
    xNext = (h/6)*(xk1 + 2*xk2 + 2*xk3 + xk4) + x
    yNext = (h/6)*(yk1 + 2*yk2 + 2*yk3 + yk4) + y
    zNext = (h/6)*(zk1 + 2*zk2 + 2*zk3 + zk4) + z
    
    return xNext, yNext, zNext
    

#Now we prepare some variables
#First the parameters
a=0.2
b=0.2
c=2.5

tList5 = np.linspace(0, h*numIter, numIter)

xTesting[0]=0
yTesting[0]=0
zTesting[0]=0
xss[0]=0
yss[0]=0
zss[0]=0

for k in range(xss.size-1):

    xTesting[k+1], yTesting[k+1], zTesting[k+1] = rosslerRK4(a, b, c, h, xTesting[k], yTesting[k], zTesting[k])
    [xss[k+1],yss[k+1],zss[k+1]]=num_rossler(xss[k],yss[k],zss[k],h,a,b,c)
#     yCurrent = xss[k+1],yss[k+1],zss[k+1]
#     xss[k+1],yss[k+1],zss[k+1]= rk4test(rossler, yCurrent, a, b, c,xss[k],yss[k],zss[k])
    
    

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(xss, yss, zss, rstride=10, cstride=10)
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)') 
plt.show()

print(xTesting)
print(xss)






