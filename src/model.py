import numpy as np

class SIR:

    def __init__(self, t_0, t_end, beta, A, d, nu, b, mu0, mu1):
        self.t_0 = t_0
        self.t_end = t_end
        self.beta = beta
        self.A = A
        self.d = d
        self.nu = nu
        self.b = b
        self.mu0 = mu0
        self.mu1 = mu1

    def mu(self, I):
        """Recovery rate.

        """
        # recovery rate, depends on mu0, mu1, b
        mu = self.mu0 + (self.mu1 - self.mu0) * (self.b / (I + self.b))
        return mu

    def R0(self):
        """
        Basic reproduction number.
        """
        return self.beta / (self.d + self.nu + self.mu1)

    def h(self, I):
        """
        Indicator function for bifurcations.
        """
        c0 = self.b ** 2 * self.d * self.A
        c1 = self.b * ((self.mu0 - self.mu1 + 2 * self.d) * self.A + (self.beta - self.nu) * self.b * self.d)
        c2 = (self.mu1 - self.mu0) * self.b * self.nu + 2 * self.b * self.d * (self.beta - self.nu) + self.d * self.A
        c3 = self.d * (self.beta - self.nu)
        res = c0 + c1 * I + c2 * I ** 2 + c3 * I ** 3
        return res

    def integration_model(self, y, t):
        S, I, R = y[:]
        m = self.mu0 + (self.mu1 - self.mu0) * (self.b / (I + self.b))

        dSdt = self.A - self.d * S - (self.beta * S * I) / (S + I + R)
        dIdt = - (self.d + self.nu) * I - m * I + (self.beta * S * I) / (S + I + R)
        dRdt = m * I - self.d * R

        return np.array([dSdt, dIdt, dRdt])
    
    def integration_model_ann(self, y, t, ann):
        
        S, I, R = y[:]
        
        m = ann.model.predict(y)
        
        #prediction_input = np.zeros((1, 3), dtype = 'float32')
        #prediction_input[0][0] = S
        #prediction_input[0][1] = I
        #prediction_input[0][2] = R

        dSdt = self.A - self.d * S - (self.beta * S * I) / (S + I + R)
        dIdt = - (self.d + self.nu) * I - m + (self.beta * S * I) / (S + I + R)
        dRdt = m - self.d * R

        return np.array([dSdt, dIdt, dRdt])



def model(t, y, mu0, mu1, beta, A, d, nu, b):
    """
    SIR model including hospitalization and natural death.

    Parameters:
    -----------
    mu0
        Minimum recovery rate
    mu1
        Maximum recovery rate
    beta
        average number of adequate contacts per unit time with infectious individuals
    A
        recruitment rate of susceptibles (e.g. birth rate)
    d
        natural death rate
    nu
        disease induced death rate
    b
        hospital beds per 10,000 persons
    """
    S, I, R = y[:]
    m = mu0 + (mu1 - mu0) * (b / (I + b))

    dSdt = A - d * S - (beta * S * I) / (S + I + R)
    dIdt = - (d + nu) * I - m * I + (beta * S * I) / (S + I + R)
    dRdt = m * I - d * R

    return [dSdt, dIdt, dRdt]
