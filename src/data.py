import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from src.model import *


def prepare_csv(file):
    df = pd.read_csv(file)

    # remove not needed columns
    df = df.drop(df.columns[[0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14]], axis=1)

    N = 125000000

    # map data to SIR model
    for index, row in df.iterrows():
        confirmed = N - row['Confirmed']
        df.at[index, 'Confirmed'] = confirmed

        fatal = row['Confirmed'] - row['Recovered'] - row['Fatal']
        df.at[index, 'Fatal'] = fatal

        recovered = row['Recovered'] + row['Fatal']
        df.at[index, 'Recovered'] = recovered

    # rename columns
    df.rename(columns={'Confirmed': 'Susceptible', 'Fatal': 'Infected'}, inplace=True)

    return df


def synthesizeSIRData(beta, a, d, nu, b, mu0, mu1, t_0, t_end, y0):
    """
    :param beta: average number of adequate contacts per unit time with infectious individuals
    :param a: birth rate of susceptible population
    :param d: per capita natural death rate
    :param nu: per capita disease-induced death rate
    :param b: number of beds per 10000 persons
    :param mu0: minimum recovery rate
    :param mu1: maximum recovery rate
    :param t_0: start time of the simulation
    :param t_end: end time of the simulation
    :param y0: starting values for the simulation
    :return: return a Pandas.DataFrame containing the values of S,I,R and the Mu-function over t_end-t0 time steps
    """
    # if these error tolerances are set too high, the solution will be qualitatively (!) wrong
    rtol = 1e-8
    atol = 1e-8

    NT = t_end - t_0
    time = np.linspace(t_0, t_end, NT)
    sol = solve_ivp(model, t_span=[time[0], time[-1]], y0=y0, t_eval=time, args=(mu0, mu1, beta, a, d, nu, b),
                    method='LSODA', rtol=rtol, atol=atol)

    df = pd.DataFrame(sol.y.transpose())

    mu = []
    for index, row in df.iterrows():
        mu.append(mu0 + (mu1 - mu0) * (b / (row[1] + b)) * row[1])
    df.insert(0, "Mu", mu)

    df.set_axis(['Mu','Susceptible', 'Infected', 'Recovered'], axis=1, inplace=True)

    return df

def createDataSet(beta, a, d, nu, b, mu0, mu1, t_0, t_end, y0, delta1, delta2):
    """
    :param beta:
    :param a:
    :param d:
    :param nu:
    :param b:
    :param mu0:
    :param mu1:
    :param t_0:
    :param t_end:
    :param y0:
    :param delta1:
    :param delta2:
    :return: Synthesize data with multiple models for different values of b. Combine these dataframes into one large 
    pandas dataframe. Train the ANN with this dataframe so that it's trained with multiple bifurcations.
    """
    df0 = synthesizeSIRData(beta, a, d, nu, b+delta1, mu0, mu1, t_0, t_end, y0)
    df1 = synthesizeSIRData(beta, a, d, nu, b+delta2, mu0, mu1, t_0, t_end, y0)
    df2 = synthesizeSIRData(beta, a, d, nu, b-delta1, mu0, mu1, t_0, t_end, y0)
    df3 = synthesizeSIRData(beta, a, d, nu, b-delta2, mu0, mu1, t_0, t_end, y0)
    df4 = synthesizeSIRData(beta, a, d, nu, b + 2*delta1, mu0, mu1, t_0, t_end, y0)
    df5 = synthesizeSIRData(beta, a, d, nu, b + 2*delta2, mu0, mu1, t_0, t_end, y0)
    df6 = synthesizeSIRData(beta, a, d, nu, b - 2*delta1, mu0, mu1, t_0, t_end, y0)
    df7 = synthesizeSIRData(beta, a, d, nu, b - 2*delta2, mu0, mu1, t_0, t_end, y0)

    return pd.concat([df0, df1, df2, df3, df4, df5, df6, df7], axis=0)

def normalizeDataSet(dataFrame):
    """
    :param dataFrame: pandas data frame to normalize
    :return: apply min-max feature scaling and return data frame
    """
    df = dataFrame.copy()
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

    return df

def generateSIRFile(df, file_name):
    """
    :param data: a list containing the values of S, I and R over a certain time
    :param file_name: the name of the output file
    :return: a file containing the values from the data parameter
    """
    df.to_csv(file_name)
