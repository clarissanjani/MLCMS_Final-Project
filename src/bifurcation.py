import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import math

"""
define system in terms of separated differential equations
d/dt x = f(x,y)
d/dt y = g(x,y)
"""

def normalform(x, y, alpha, mu0, mu1, b, I, d):
    """
    define normal form based on SIR model implementation of Lu and Huang. 
    :param x: 
    :param y: 
    :param alpha: psychological effect as defined by paper
    """
    # hard coded values
    mu = 0.5 # mu is greater than 0 and is the natural recovery rate of infective individuals
    lam = 0.5  # lambda is greater than 0 and is the rate at which recovered individuals lose immunity
    k = 0.5 # k is the infection rate
    m = mu0 + (mu1 - mu0) * (b / (I + b))
    beta = -2 * math.sqrt(alpha) # beta is greater than the square root of alpha multipled with -2
    
    n = alpha * ((d + lam) / k)
    p = (d + mu) / (d + lam) 
    a = (p - 1 - n*p) / (n +m+1)
    q = mu / (d + lam)
    b_tilda = p - 1 - n * p
    w = math.sqrt(abs(b_tilda * q - b_tilda * b_tilda))
    a_20 = - p / w
    a_11 = 2
    a_2 = 0
    a_30 = -p / w
    a_21 = 1 
    a_12 = 0 
    a_3 = 0
    
    b_20 = (1 / w**2) * ( - b_tilda * p + a * (m + 2*n) * (b_tilda - q))
    b_11 = (2 * b_tilda - a * (m+2 * n))
    b_30 = - (1 / w * w) * (b_tilda * p + a * (m + 2 * n) * (b_tilda - q))
    b_21 = (b_tilda - a * n) / w
    b_12 = 0
    b_3 = 0
    b_2 = 0
    
    # known as the u or f(x,y)
    f = a_20 * x ** 2 + a_11 * x * y + a_2 * y ** 2 + a_30 * x ** 3 + a_21 * x ** 2 * y + a_12 * x * y ** 2 + a_3 * y ** 3
    # known as the v or g(x,y)
    g = b_20 * x ** 2 + b_11 * x * y + b_2 * y ** 2 + b_30 * x ** 3 + b_21 * y * x ** 2 + b_12 * x * y ** 2 + b_3 * y ** 3

    return [f, g]