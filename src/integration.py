import numpy as np


def euler(integration_model, y0, t):
    """
    :param integration_model: ODEs from the SIR model
    :param y0: initial value
    :param t: time axis
    :return: integrated values of S, I and R
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


def euler_ann(integration_model, y0, t, ann):
    """
    Same as Euler but calls the SIR model with the approximated
    function f from the Artifi
    :param integration_model:
    :param y0:
    :param t:
    :param ann: Artificial Neural Network to approximate mu(b,I) in the SIR model
    :return:
    """
    s = np.zeros(len(t))
    i = np.zeros(len(t))
    r = np.zeros(len(t))

    s[0], i[0], r[0] = y0

    for n in range(0, len(t) - 1):
        
        y = [s[n], i[n], r[n]]
        
        y_next = y + np.multiply(integration_model(y, t[n], ann), (t[n + 1] - t[n]))
        
        s[n+1] = y_next[0]
        i[n+1] = y_next[1]
        r[n+1] = y_next[2]

    return s, i, r


def classic_runge_kutta(integration_model, y0, t):
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
    
    for n in range (0, len(t) - 1):
        
        y = [s[n], i[n], r[n]]
        h = t[1] - t[0]        
        k_1 = integration_model(y, t[n])
        # print("k_1: ", k_1)
        k_2 = integration_model(y + (h/2)*k_1, t[n] + (h/2))
        # print("k_2: ", k_2)
        k_3 = integration_model(y + (h/2)*k_2, t[n] + (h/2))
        # print("k_3: ", k_3)
        k_4 = integration_model(y + h*k_3, t[n] + h)
        # print("k_4: ", k_4)

        
        y_next = y + (h/6) * (k_1 + 2*k_2 + 2*k_3 + k_4)
        
        s[n+1] = y_next[0]
        i[n+1] = y_next[1]
        r[n+1] = y_next[2]
        
    return s, i, r


def classic_runge_kutta_ann(integration_model, y0, t, ann):
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
    
    for n in range (0, len(t) - 1):
        
        y = [s[n], i[n], r[n]]
        h = t[1] - t[0]        
        
        k_1 = integration_model(y, t[n], ann)
        k_2 = integration_model(y + (h/2)*k_1, t[n] + (h/2), ann)
        k_3 = integration_model(y + (h/2)*k_2, t[n] + (h/2), ann)
        k_4 = integration_model(y + h*k_3, t[n] + h, ann)
        
        y_next = y + (h/6) * (k_1 + 2*k_2 + 2*k_3 + k_4)
        
        s[n+1] = y_next[0]
        i[n+1] = y_next[1]
        r[n+1] = y_next[2]

    return s, i, r

