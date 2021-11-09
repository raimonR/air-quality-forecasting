import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mape, mse
import torch

# TODO: Vary the number of blocks to determine optimal value
# TODO: Vary the layer width to determine optimal value
# TODO: Vary the number of epochs
# TODO: Vary the learning rate
model = NBEATSModel(input_chunk_length=24, output_chunk_length=24, generic_architecture=False, num_blocks=3,
                    layer_widths=512, n_epochs=300, optimizer_cls=torch.optim.Adam,
                    optimizer_kwargs={'lr': 0.1, 'eps': 1e-7},
                    lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    lr_scheduler_kwargs={'factor': 0.2, 'patience': 10, 'min_lr': 0.001}, model_name='base_model',
                    force_reset=True)

files = os.listdir('dataset/transfer_learning/')
for f in [files[0]]:
    train_set = pd.read_pickle(f'dataset/transfer_learning/{f}/train_set.pkl')
    dev_set = pd.read_pickle(f'dataset/transfer_learning/{f}/dev_set.pkl')
    test_set = pd.read_pickle(f'dataset/transfer_learning/{f}/test_set.pkl')

    # n = test_forecast.shape[0]

    train_forecast = TimeSeries.from_dataframe(train_set, value_cols='pm25', freq='1H')
    dev_forecast = TimeSeries.from_dataframe(dev_set, value_cols='pm25', freq='1H')
    test_forecast = TimeSeries.from_dataframe(test_set, value_cols='pm25', freq='1H')

    train_covar = TimeSeries.from_dataframe(train_set, value_cols='LATITUDE', freq='1H')
    dev_covar = TimeSeries.from_dataframe(dev_set, value_cols='LATITUDE', freq='1H')
    test_covar = TimeSeries.from_dataframe(test_set, value_cols='LATITUDE', freq='1H')
    columns = train_set.columns[(train_set.columns != 'pm25') & (train_set.columns != 'LATITUDE')].to_list()
    for column in columns:
        train_covar = train_covar.stack(TimeSeries.from_dataframe(train_set, value_cols=column, freq='1H'))
        dev_covar = dev_covar.stack(TimeSeries.from_dataframe(dev_set, value_cols=column, freq='1H'))
        test_covar = test_covar.stack(TimeSeries.from_dataframe(test_set, value_cols=column, freq='1H'))

    model.fit(series=train_forecast, past_covariates=train_covar, val_series=dev_forecast,
              val_past_covariates=dev_covar)

    forecast = model.predict(n=n, series=test_forecast, past_covariates=test_covar)

    fig, ax = plt.subplots()
    ax.plot(test_forecast.data_array().to_numpy())
    ax.plot(forecast.values)
    fig.savefig('test_plot_for_nbeats_model.png')
    plt.close()
