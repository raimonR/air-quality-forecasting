import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from itertools import zip_longest
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mape, mse
import torch
    

# TODO: Vary the number of blocks to determine optimal value
# TODO: Vary the layer width to determine optimal value
# TODO: Vary the number of epochs
# TODO: Vary the learning rate
model = NBEATSModel(input_chunk_length=48, output_chunk_length=24, generic_architecture=False, num_blocks=1,
                    layer_widths=1, n_epochs=300, optimizer_cls=torch.optim.Adam,
                    optimizer_kwargs={'lr': 0.01, 'eps': 1e-7}, model_name='base_model', force_reset=True)

files = os.listdir('dataset/transfer_learning/')
for idx, f in enumerate([files[0]]):
    train_sets = os.listdir(f'dataset/transfer_learning/{f}/train_sets/')
    dev_sets = os.listdir(f'dataset/transfer_learning/{f}/dev_sets/')
    test_sets = os.listdir(f'dataset/transfer_learning/{f}/test_sets/')

    n = np.floor(len(train_sets)/len(dev_sets)).astype(int)
    dev_sets = np.concatenate([dev_sets for i in range(n)]).tolist()
    for i in range(len(train_sets) % len(dev_sets)):
        dev_sets.append(dev_sets[i])

    zip_sets = list(zip_longest(train_sets, dev_sets))
    t0 = time.perf_counter()
    for sets in [zip_sets[0]]:
        train_set = pd.read_pickle(f'dataset/transfer_learning/{f}/train_sets/{sets[0]}')
        dev_set = pd.read_pickle(f'dataset/transfer_learning/{f}/dev_sets/{sets[1]}')

        train_forecast = TimeSeries.from_dataframe(train_set, value_cols='pm25', freq='1H')
        train_forecast = TimeSeries.from_xarray(train_forecast.data_array().astype('float32'))
        dev_forecast = TimeSeries.from_dataframe(dev_set, value_cols='pm25', freq='1H')
        dev_forecast = TimeSeries.from_xarray(dev_forecast.data_array().astype('float32'))

        train_covar = TimeSeries.from_dataframe(train_set, value_cols='LATITUDE', freq='1H')
        train_covar = TimeSeries.from_xarray(train_covar.data_array().astype('float32'))
        dev_covar = TimeSeries.from_dataframe(dev_set, value_cols='LATITUDE', freq='1H')
        dev_covar = TimeSeries.from_xarray(dev_covar.data_array().astype('float32'))

        columns = train_set.columns[(train_set.columns != 'pm25') & (train_set.columns != 'LATITUDE')].to_list()
        for column in columns:
            train_temp = TimeSeries.from_dataframe(train_set, value_cols=column, freq='1H')
            train_temp = TimeSeries.from_xarray(train_temp.data_array().astype('float32'))
            train_covar = train_covar.stack(train_temp)

            dev_temp = TimeSeries.from_dataframe(dev_set, value_cols=column, freq='1H')
            dev_temp = TimeSeries.from_xarray(dev_temp.data_array().astype('float32'))
            dev_covar = dev_covar.stack(dev_temp)

        model.fit(series=train_forecast, past_covariates=train_covar, val_series=dev_forecast,
                  val_past_covariates=dev_covar, verbose=True)

    predictions_array = np.array([])
    true_array = np.array([])
    normalizer_y = load(f'dataset/transfer_learning/{f}/normalizer_y.joblib')
    os.makedirs(f'results/tests/nbeats/{f}/', exist_ok=True)
    for sets in [test_sets[0]]:
        test_set = pd.read_pickle(f'dataset/transfer_learning/{f}/test_sets/{sets}')

        test_forecast = TimeSeries.from_dataframe(test_set, value_cols='pm25', freq='1H')
        test_forecast = TimeSeries.from_xarray(test_forecast.data_array().astype('float32'))
        test_covar = TimeSeries.from_dataframe(test_set, value_cols='LATITUDE', freq='1H')
        test_covar = TimeSeries.from_xarray(test_covar.data_array().astype('float32'))

        columns = test_set.columns[(test_set.columns != 'pm25') & (test_set.columns != 'LATITUDE')].to_list()
        for column in columns:
            test_temp = TimeSeries.from_dataframe(test_set, value_cols=column, freq='1H')
            test_temp = TimeSeries.from_xarray(test_temp.data_array().astype('float32'))
            test_covar = test_covar.stack(test_temp)

        forecast = model.historical_forecasts(series=test_forecast, past_covariates=test_covar, start=48,
                                              forecast_horizon=24, stride=1, retrain=False, verbose=True)

        forecast = forecast.pd_series()
        true_y = test_forecast.pd_series()
        fig, ax = plt.subplots()
        ax.plot(true_y.index.to_numpy(), true_y.values, label=r'$y$')
        ax.plot(forecast.index.to_numpy(), forecast.values, label=r'$\hat{y}$')
        ax.set(xlabel='Date', ylabel=r'Normalized $PM_{2.5}$')
        # plt.show()
        fig.savefig(f'results/tests/nbeats/{f}/test_plot_for_nbeats_model.png')
        plt.close()
