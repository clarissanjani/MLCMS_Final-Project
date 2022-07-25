import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from bifurcation import *

"""
define system in terms of separated differential equations
d/dt x = f(x,y)
d/dt y = g(x,y)
"""


def f(x, y, alpha):
    return alpha * x - y - x * (x ** 2 + y ** 2)


def g(x, y, alpha):
    return x + alpha * y - y * (x ** 2 + y ** 2)

def h(x, y, alpha):
    return x + alpha * y - y * (x ** 2 + y ** 2)


"""
 define system in terms of a Numpy array
"""


def Sys(X, t=0, alpha=1):
    # here X[0] = x and x[1] = y    
    dx = alpha * X[0] - X[1] - X[0] * (X[0] ** 2 + X[1] ** 2)
    dy = X[0] + alpha * X[1] - X[1] * (X[0] ** 2 + X[1] ** 2)
    return np.array([dx, dy])


"""
ode for the solve_ivp function, special order of arguments
"""
def ode(t, state, alpha):
    x, y = state
    
    dx = alpha * x - y - x * (x ** 2 + y ** 2)
    dy = x + alpha * y - y * (x ** 2 + y ** 2)
    
    return dx, dy


"""
Find fixed point (f(x,y) = 0 and g(x,y) = 0 by brute force
"""


def find_fixed_points(r, f, g, alpha):
    fp = []
    for x in range(r):
        for y in range(r):
            if ((f(x, y, alpha) == 0) and (g(x, y, alpha) == 0)):
                fp.append((x, y))
                print('The system has a fixed point in %s,%s' % (x, y))
    return fp
