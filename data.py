import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from model import *
import csv


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
    :return: a list containing the values of S,I and R over t_end-t0 time steps
    """
    # if these error tolerances are set too high, the solution will be qualitatively (!) wrong
    rtol = 1e-8
    atol = 1e-8

    NT = t_end - t_0
    time = np.linspace(t_0, t_end, NT)
    sol = solve_ivp(model, t_span=[time[0], time[-1]], y0=y0, t_eval=time, args=(mu0, mu1, beta, a, d, nu, b),
                    method='LSODA', rtol=rtol, atol=atol)
    return sol


def generateSIRFile(data, file_name):
    """
    :param data: a list containing the values of S, I and R over a certain time
    :param file_name: the name of the output file
    :return: a file containing the values from the data parameter
    """
    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Susceptible', 'Infected', 'Recovered'])
        writer.writerows(data.transpose())
