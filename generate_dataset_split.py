import pandas as pd
import numpy as np
import os
from darts.timeseries import TimeSeries
from sklearn.preprocessing import StandardScaler
import json


def individual_dataset_split():
    files = os.listdir('dataset/merged/')
    for f in files:
        set_x = []
        set_y = []
        df_temp = pd.read_pickle(f'dataset/merged/{f}')
        df_temp = df_temp.fillna(0)

        for e in df_temp.columns.to_list()[:-450]:
            if e not in ('LATITUDE', 'LONGITUDE', 'ELEVATION'):
                df_temp[e] = StandardScaler().fit_transform(df_temp[e].to_numpy().reshape(-1, 1))

        for index, row in df_temp.iterrows():
            pressure_data = row[-450:-300].to_numpy()
            pressure_data = StandardScaler().fit_transform(pressure_data.reshape(-1, 1))
            df_temp.loc[index][-450:-300] = pressure_data.reshape(150, )

            temperature_data = row[-300:-150].to_numpy()
            temperature_data = StandardScaler().fit_transform(temperature_data.reshape(-1, 1))
            df_temp.loc[index][-300:-150] = temperature_data.reshape(150, )

            thetav_data = row[-150:].to_numpy()
            thetav_data = StandardScaler().fit_transform(thetav_data.reshape(-1, 1))
            df_temp.loc[index][-150:] = thetav_data.reshape(150, )

        num = df_temp.shape[0]
        split_1 = int(num*0.8)
        split_2 = int(num*0.9)
        train_set_x, train_set_y = df_temp.iloc[:split_1, 1:].to_numpy(), df_temp[:split_1]['pm25'].to_numpy()
        dev_set_x, dev_set_y = df_temp.iloc[split_1:split_2, 1:].to_numpy(), df_temp[split_1:split_2]['pm25'].to_numpy()
        test_set_x, test_set_y = df_temp.iloc[split_2:, 1:].to_numpy(), df_temp[split_2:]['pm25'].to_numpy()

        np.save(f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/train_set_x', train_set_x)
        np.save(f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/train_set_y', train_set_y)
        np.save(f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/dev_set_x', dev_set_x)
        np.save(f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/dev_set_y', dev_set_y)
        np.save(f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/test_set_x', test_set_x)
        np.save(f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/test_set_y', test_set_y)


def grouped_dataset_split(rng_int: int):
    rng = np.random.default_rng(rng_int)
    files = os.listdir('dataset/merged/')
    set_x = []
    set_y = []
    for f in files:
        df_temp = pd.read_pickle(f'dataset/merged/{f}')
        df_temp = df_temp.fillna(0)

        for e in df_temp.columns.to_list()[:-450]:
            if e not in ('LATITUDE', 'LONGITUDE', 'ELEVATION'):
                df_temp[e] = StandardScaler().fit_transform(df_temp[e].to_numpy().reshape(-1, 1))

        for index, row in df_temp.iterrows():
            pressure_data = row[-450:-300].to_numpy()
            pressure_data = StandardScaler().fit_transform(pressure_data.reshape(-1, 1))
            df_temp.loc[index][-450:-300] = pressure_data.reshape(150, )

            temperature_data = row[-300:-150].to_numpy()
            temperature_data = StandardScaler().fit_transform(temperature_data.reshape(-1, 1))
            df_temp.loc[index][-300:-150] = temperature_data.reshape(150, )

            thetav_data = row[-150:].to_numpy()
            thetav_data = StandardScaler().fit_transform(thetav_data.reshape(-1, 1))
            df_temp.loc[index][-150:] = thetav_data.reshape(150, )

        min_index = df_temp.index[0]
        if min_index.hour != 0:
            min_index = df_temp.index[df_temp.index.hour == 0][0]

        max_index = df_temp.index[-1]
        if max_index.hour != 0:
            max_index = df_temp.index[df_temp.index.hour == 0][-1]

        cal = np.arange(min_index, max_index, step=np.timedelta64(1, 'D'), dtype='datetime64')
        n = np.floor(cal.shape[0]/2*0.9).astype(int)
        cal = rng.choice(cal, n, replace=False, shuffle=False)
        for day in cal:
            if df_temp.loc[day:(day + np.timedelta64(2, 'D')), :].shape[0] < 48:
                continue
            set_x.append(df_temp.loc[day:(day + np.timedelta64(23, 'h')), :])
            set_y.append(df_temp.loc[(day + np.timedelta64(1, 'D')):(day + np.timedelta64(47, 'h')), 'pm25'].values)

        print(f'Approx. number of {f.split("_")[0]} elements: {n}')

    set_x = np.array(set_x)
    set_y = np.array(set_y)
    set_y = set_y.reshape(set_y.shape[0], set_y.shape[1], 1)

    num = set_x.shape[0]
    shuffle = rng.permutation(num)
    set_x = set_x[shuffle, :, :]
    set_y = set_y[shuffle, :, :]

    split_1 = int(num*0.8)
    split_2 = int(num*0.9)

    train_set_x, train_set_y = set_x[:split_1], set_y[:split_1]
    dev_set_x, dev_set_y = set_x[split_1:split_2], set_y[split_1:split_2]
    test_set_x, test_set_y = set_x[split_2:], set_y[split_2:]

    np.save('dataset/lstm_dataset_splits/collective/train_set_x', train_set_x)
    np.save('dataset/lstm_dataset_splits/collective/train_set_y', train_set_y)
    np.save('dataset/lstm_dataset_splits/collective/dev_set_x', dev_set_x)
    np.save('dataset/lstm_dataset_splits/collective/dev_set_y', dev_set_y)
    np.save('dataset/lstm_dataset_splits/collective/test_set_x', test_set_x)
    np.save('dataset/lstm_dataset_splits/collective/test_set_y', test_set_y)


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

        # TODO: FIND A WAY TO STORE THE COVARIATES AND FORECAST IN MORE SPACE EFFICIENT FILES
        with open(f'./dataset/transfer_learning/{f.split("_")[0]}/forecast.json', 'w') as write_file:
            json.dump(forecast.to_json(), write_file)

        with open(f'./dataset/transfer_learning/{f.split("_")[0]}/covariates.json', 'w') as write_file:
            json.dump(covariates.to_json(), write_file)


individual_dataset_split()
# grouped_dataset_split(0)
# transfer_dataset_split()
