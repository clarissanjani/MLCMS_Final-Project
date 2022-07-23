import numpy as np


def euler(integration_model, y0, t):
    """
    :param integration_model:
    :param y0:
    :param t:
    :return:
    """
    s = np.zeros(len(t))
    i = np.zeros(len(t))
    r = np.zeros(len(t))

    s[0], i[0], r[0] = y0

    for n in range(0, len(t) - 1):
        y = [s[n], i[n], r[n]]
        y_next = y + np.multiply(integration_model(y, t[n]), (t[n + 1] - t[n]))
        s[n+1] = y_next[0]
        i[n+1] = y_next[1]
        r[n+1] = y_next[2]

    return s, i, r


