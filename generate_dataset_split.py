import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from darts import TimeSeries
import json


def grouped_dataset_split(rng_int: int):
    rng = np.random.default_rng(rng_int)
    files = os.listdir('dataset/merged/')
    for f in files:
        if f == 'Abidjan_merged_dataset.pkl':
            df_temp = pd.read_pickle(f'dataset/merged/{f}')
            df_temp = df_temp.fillna(0)

            scaler = StandardScaler()
            scaled_array = scaler.fit_transform(df_temp)
            for i, c in enumerate(df_temp.columns):
                df_temp[c] = scaled_array[:, i]

            min_index = df_temp.index[0]
            if min_index.hour != 0:
                min_index = df_temp.index[df_temp.index.hour == 0][0]

            max_index = df_temp.index[-1]
            if max_index.hour != 0:
                max_index = df_temp.index[df_temp.index.hour == 0][-1]

            abidjan_cal = np.arange(min_index, max_index, step=np.timedelta64(1, 'D'), dtype='datetime64')
            n = np.floor(abidjan_cal.shape[0]/2*0.9).astype(int)
            abidjan_cal = rng.choice(abidjan_cal, n, replace=False, shuffle=False)
            sub_sample = []
            for day in abidjan_cal:
                sub_sample.append(df_temp.loc[day:(day + np.timedelta64(2, 'D')), :])

            print('asdf')

        elif f == 'Chennai_merged_dataset.pkl':
            df_temp = pd.read_pickle(f'dataset/merged/{f}')
            df_temp = df_temp.fillna(0)

            scaler = StandardScaler()
            scaled_array = scaler.fit_transform(df_temp)
            for i, c in enumerate(df_temp.columns):
                df_temp[c] = scaled_array[:, i]

            min_index = df_temp.index[0]
            max_index = df_temp.index[-1]
            chennai_cal = np.arange(min_index, max_index, step=np.timedelta64(1, 'D'), dtype='datetime64')
        elif f == 'Melbourne_merged_dataset.pkl':
            df_temp = pd.read_pickle(f'dataset/merged/{f}')
            df_temp = df_temp.fillna(0)

            scaler = StandardScaler()
            scaled_array = scaler.fit_transform(df_temp)
            for i, c in enumerate(df_temp.columns):
                df_temp[c] = scaled_array[:, i]

            min_index = df_temp.index[0]
            max_index = df_temp.index[-1]
            melbourne_cal = np.arange(min_index, max_index, step=np.timedelta64(1, 'D'), dtype='datetime64')
        elif f == 'Munich_merged_dataset.pkl':
            df_temp = pd.read_pickle(f'dataset/merged/{f}')
            df_temp = df_temp.fillna(0)

            scaler = StandardScaler()
            scaled_array = scaler.fit_transform(df_temp)
            for i, c in enumerate(df_temp.columns):
                df_temp[c] = scaled_array[:, i]

            min_index = df_temp.index[0]
            max_index = df_temp.index[-1]
            munich_cal = np.arange(min_index, max_index, step=np.timedelta64(1, 'D'), dtype='datetime64')
        elif f == 'Oakland_merged_dataset.pkl':
            df_temp = pd.read_pickle(f'dataset/merged/{f}')
            df_temp = df_temp.fillna(0)

            scaler = StandardScaler()
            scaled_array = scaler.fit_transform(df_temp)
            for i, c in enumerate(df_temp.columns):
                df_temp[c] = scaled_array[:, i]

            min_index = df_temp.index[0]
            max_index = df_temp.index[-1]
            oakland_cal = np.arange(min_index, max_index, step=np.timedelta64(1, 'D'), dtype='datetime64')
        elif f == 'Santiago_merged_dataset.pkl':
            df_temp = pd.read_pickle(f'dataset/merged/{f}')
            df_temp = df_temp.fillna(0)

            scaler = StandardScaler()
            scaled_array = scaler.fit_transform(df_temp)
            for i, c in enumerate(df_temp.columns):
                df_temp[c] = scaled_array[:, i]

            min_index = df_temp.index[0]
            max_index = df_temp.index[-1]
            santiago_cal = np.arange(min_index, max_index, step=np.timedelta64(1, 'D'), dtype='datetime64')


def transfer_dataset_split():
    files = os.listdir('dataset/merged/')
    for f in files:
        df_temp = pd.read_pickle(f'dataset/merged/{f}')
        df_temp = df_temp.fillna(0)

        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df_temp)
        for i, c in enumerate(df_temp.columns):
            df_temp[c] = scaled_array[:, i]

        forecast = TimeSeries.from_dataframe(df_temp, value_cols='pm25', freq='1H')

        covariates = TimeSeries.from_dataframe(df_temp, value_cols='LATITUDE', freq='1H')
        columns = df_temp.columns[(df_temp.columns != 'pm25') & (df_temp.columns != 'LATITUDE')].to_list()
        for column in columns:
            covariates = covariates.stack(TimeSeries.from_dataframe(df_temp, value_cols=column, freq='1H'))

        with open(f'./dataset/transfer_learning/{f.split("_")[0]}/forecast.json', 'w') as write_file:
            json.dump(forecast.to_json(), write_file)

        with open(f'./dataset/transfer_learning/{f.split("_")[0]}/covariates.json', 'w') as write_file:
            json.dump(covariates.to_json(), write_file)


grouped_dataset_split(0)
# transfer_dataset_split()
