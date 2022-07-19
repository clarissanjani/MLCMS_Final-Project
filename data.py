import pandas as pd


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
