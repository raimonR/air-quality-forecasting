import numpy as np
from numpy.random import default_rng
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import impute


def combined_datasets():
    # Define random seed for numpy
    rng = default_rng(0)

    # Number of preceding days for AQ forecasting
    d = 1


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
                aq_data.asfreq(freq='1H')
                aq_data[aq_data < 0] = np.nan
                imp = impute.KNNImputer(n_neighbors=10)
                aq_data = imp.fit_transform(aq_data.values)

                merged_df.append(aq_data)

            elif f == 'daily_weather_postprocessed':
                weather_data = pd.read_pickle(f'./dataset/{f}/{city}')
                weather_data = weather_data.set_index('DATE')
                weather_data.index = pd.to_datetime(weather_data.index, utc=False)
                weather_data[weather_data['SNDP'].isna()] = 0
                weather_data[weather_data['sea level pressure'].isna()] = 1013.25
                weather_data[weather_data['precipitation'].isna()] = 0
                for cols in weather_data.columns:
                    weather_data[weather_data[cols].isna()] = weather_data[cols].mean()

                # TODO: iteratively remove weather inputs to determine which ones are relevant
                #  specifically: 'sea level pressure', 'GUST', 'visibility', 'weather'
                merged_df.append(weather_data)

            elif f == 'drought_postprocessed':
                drought_data = pd.read_pickle(f'./dataset/{f}/{city}')
                drought_data = drought_data.set_index('Date')
                drought_data.index = pd.to_datetime(drought_data.index, utc=False)

                merged_df.append(drought_data)

            elif f == 'radiosonde_postprocessed':
                radiosonde_data = pd.read_pickle(f'./dataset/{f}/{city}')

                print('a')




individual_datasets()
