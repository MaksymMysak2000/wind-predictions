import pandas as pd
import numpy as np


# read csv file with meteodata
def read_wind_meteodata():

    csv_path = 'data/one_season_one_location.csv'

    df = pd.read_csv(csv_path)
    df.index = pd.to_datetime(df['date'], format='%Y.%m.%d %H:%M:%S')

    wind = df['wdsp']

    return wind


# prepare input/output values
def df_to_X_y(df, window_size=5):

    df_as_np = df.to_numpy()
    X = []
    y = []

    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i:i + window_size]]
        X.append(row)
        label = df_as_np[i + window_size]
        y.append(label)

    return np.array(X), np.array(y)