import numpy as np
from numpy.random import default_rng
import pandas as pd
import matplotlib.pyplot as plt
import os


def combined_datasets():
    # Define random seed for numpy
    rng = default_rng(0)

    # Number of preceding days for AQ forecasting
    d = 1

# TODO: DETERMINE METHOD FOR REPLACING MISSING DATA
def individual_datasets():
    # Define random seed for numpy
    rng = default_rng(0)

    # Number of preceding days for AQ forecasting
    d = 1

    # Get datasets for each city
    for city in ['Accra.pkl', 'Chennai.pkl', 'Melbourne.pkl', 'Santiago.pkl', 'SF.pkl', 'Toulouse.pkl']:
        ds = ['air_quality_data', 'daily_weather_postprocessed', 'drought_postprocessed', 'radiosonde_postprocessed']
        merged_df = []
        for f in ds:
            if f == 'air_quality_data':
                aq_data = pd.read_pickle(f'./dataset/{f}/{city}.zip')
                aq_data = aq_data.set_index('date')
                aq_data.index = pd.to_datetime(aq_data.index, utc=False)
                aq_data = aq_data.fillna(value=-999)
                aq_data.asfreq(freq='1H', fill_value=-999)

                merged_df.append(aq_data)

            elif f == 'daily_weather_postprocessed':
                weather_data = pd.read_pickle(f'./dataset/{f}/{city}')
                weather_data = weather_data.set_index('DATE')
                weather_data.index = pd.to_datetime(weather_data.index, utc=False)
                weather_data.iloc[:, np.r_[0:14, -1]] = weather_data.iloc[np.r_[0:14, -1]].fillna(
                                                            value=weather_data.iloc[:, np.r_[0:14, -1]].mean())

                print('a')

            elif f == 'drought_postprocessed':
                drought_data = pd.read_pickle(f'./dataset/{f}/{city}')

            elif f == 'radiosonde_postprocessed':
                radiosonde_data = pd.read_pickle(f'./dataset/{f}/{city}')




individual_datasets()
