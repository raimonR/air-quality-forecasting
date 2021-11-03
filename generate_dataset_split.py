import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
# TODO: SAVE SCALERS SO THAT DATA CAN BE CORRECTLY RESCALED AND COMPARED
# TODO: determine the correct procedure for scaling inputs and outputs or only inputs


# TODO: this data needs to be reshaped into sequential set.
def individual_dataset_split(normalize: bool):
    from tensorflow.keras.preprocessing import timeseries_dataset_from_array
    from tensorflow import data, autograph

    autograph.set_verbosity(0)
    files = os.listdir('dataset/merged/')
    for f in files:
        df_temp = pd.read_pickle(f'dataset/merged/{f}')
        df_temp = df_temp.fillna(0)

        if normalize:
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

        def dataset_generator(inputs, output_dims, input_length, output_length, batch_size):
            dataset = timeseries_dataset_from_array(inputs, None, input_length + output_length, batch_size=batch_size)

            def split_inputs(x):
                return x[:, :input_length, :], x[:, input_length:, output_dims]

            dataset = dataset.map(split_inputs)

            return dataset

        num = df_temp.shape[0]
        split_1 = int(num*0.8)
        split_2 = int(num*0.9)
        train_set = df_temp.iloc[:split_1, :].to_numpy()
        dev_set = df_temp.iloc[split_1:split_2, 1:].to_numpy()
        test_set = df_temp.iloc[split_2:, 1:].to_numpy()
        train_set = dataset_generator(train_set, slice(0, 1), input_length=24, output_length=24, batch_size=128)
        dev_set = dataset_generator(dev_set, slice(0, 1), input_length=24, output_length=24, batch_size=128)
        test_set = dataset_generator(test_set, slice(0, 1), input_length=24, output_length=24, batch_size=128)

        data.experimental.save(train_set, f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/train_set')
        data.experimental.save(dev_set, f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/dev_set')
        data.experimental.save(test_set, f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/test_set')


# TODO: Testing if unscaled inputs work as well
def grouped_dataset_split(rng_int: int, normalize: bool):
    rng = np.random.default_rng(rng_int)
    files = os.listdir('dataset/merged/')
    set_x = []
    set_y = []
    for f in files:
        df_temp = pd.read_pickle(f'dataset/merged/{f}')
        df_temp = df_temp.fillna(0)

        if normalize:
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
            if df_temp.loc[day:(day + np.timedelta64(4, 'D')), :].shape[0] < 96:
                continue
            set_x.append(df_temp.loc[day:(day + np.timedelta64(71, 'h')), :])
            set_y.append(df_temp.loc[(day + np.timedelta64(3, 'D')):(day + np.timedelta64(95, 'h')), 'pm25'].values)

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
    from darts.timeseries import TimeSeries
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

        forecast_df = forecast.data_array().to_dataframe('forecast')
        forecast_df.to_pickle(f'./dataset/transfer_learning/{f.split("_")[0]}/forecast.pkl')

        covariates_df = covariates.data_array().to_dataframe('covariates')
        covariates_df.to_pickle(f'./dataset/transfer_learning/{f.split("_")[0]}/covariates.pkl')


individual_dataset_split(True)
grouped_dataset_split(0, True)
# transfer_dataset_split()
