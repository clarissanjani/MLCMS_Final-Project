import numpy as np


def euler(f, y0, t):
    """

    """
    y = np.zeros(len(t))
    y[0] = y0
    for n in range(0, len(t) - 1):
        y[n+1] = y[n] + f(y[n], t[n]) * (t[n+1] - t[n])
    return y


