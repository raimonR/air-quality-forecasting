import os
import time
import csv
from joblib import load
from itertools import zip_longest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Bidirectional
from keras.regularizers import l1_l2
from tensorflow import autograph

autograph.set_verbosity(0)


@tf.autograph.experimental.do_not_convert
def generate_inputs_outputs(data, n_past, n_horizon, batch_num, shift):
    def make_batch(x):
        return x.batch(length, drop_remainder=True)

    def make_split(x):
        return x[:-n_horizon], x[-n_horizon:, 0]

    length = n_past + n_horizon
    ds = tf.data.Dataset.from_tensor_slices(data)

    ds = ds.window(length, shift=shift, drop_remainder=True)
    ds = ds.flat_map(make_batch)

    ds = ds.map(make_split)

    ds = ds.batch(batch_num, drop_remainder=True)
    return ds


# Define hyperparameters and other parameters
epochs = 1000
learning_rate = 1e-2
l1l2 = (0.0, 0.0)
n_features = 468
past = 48
horizon = 24
batch_numbers = [128]*9

opt = keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=100, min_lr=0.0001)

model = Sequential()
model.add(Input(shape=(past, n_features)))
model.add(Bidirectional(LSTM(units=128, return_sequences=True, activity_regularizer=l1_l2(l1l2[0], l1l2[1]))))
model.add(Bidirectional(LSTM(units=128, return_sequences=True, activity_regularizer=l1_l2(l1l2[0], l1l2[1]))))
model.add(Bidirectional(LSTM(units=128, return_sequences=True, activity_regularizer=l1_l2(l1l2[0], l1l2[1]))))
model.add(Bidirectional(LSTM(units=64, kernel_regularizer=l1_l2(l1l2[0], l1l2[1]))))
model.add(Dense(units=horizon))
model.summary()

model.compile(optimizer=opt, loss='mse')

files = os.listdir('dataset/lstm_dataset_splits/individual/')
for idx, f in enumerate(files):
    train_sets = os.listdir(f'dataset/lstm_dataset_splits/individual/{f}/train_sets/')
    dev_sets = os.listdir(f'dataset/lstm_dataset_splits/individual/{f}/dev_sets/')
    test_sets = os.listdir(f'dataset/lstm_dataset_splits/individual/{f}/test_sets/')

    n = np.floor(len(train_sets)/len(dev_sets)).astype(int)
    dev_sets = np.concatenate([dev_sets for i in range(n)]).tolist()
    for i in range(len(train_sets) % len(dev_sets)):
        dev_sets.append(dev_sets[i])

    zip_sets = list(zip_longest(train_sets, dev_sets))
    os.makedirs(f'results/tests/individual_lstm/{f}/plots/', exist_ok=True)
    t0 = time.perf_counter()
    for sets in zip_sets:
        train_set = pd.read_pickle(f'dataset/lstm_dataset_splits/individual/{f}/train_sets/{sets[0]}').to_numpy()
        dev_set = pd.read_pickle(f'dataset/lstm_dataset_splits/individual/{f}/dev_sets/{sets[1]}').to_numpy()

        train_ds = generate_inputs_outputs(train_set, past, horizon, batch_numbers[idx], 24)
        dev_ds = generate_inputs_outputs(dev_set, past, horizon, batch_numbers[idx], 24)

        i = 0
        while len(list(train_ds)) < 1:
            train_ds = generate_inputs_outputs(train_set, past, horizon, batch_numbers[idx] - i, 24)
            i += 1

        i = 0
        while len(list(dev_ds)) < 1:
            dev_ds = generate_inputs_outputs(dev_set, past, horizon, batch_numbers[idx] - i, 24)
            i += 1

        res = model.fit(x=train_ds, validation_data=dev_ds, epochs=epochs, shuffle=False,
                        callbacks=[early_stopping, reduce_lr])

        t1 = time.perf_counter()
        print(f'Time for {early_stopping.stopped_epoch} epochs:', t1 - t0)

    predictions_array = np.array([])
    true_array = np.array([])
    normalizer_y = load(f'dataset/lstm_dataset_splits/individual/{f}/normalizer_y.joblib')
    for sets in test_sets:
        test_set = pd.read_pickle(f'dataset/lstm_dataset_splits/individual/{f}/test_sets/{sets}').to_numpy()
        test_ds = generate_inputs_outputs(test_set, past, horizon, 128, 24)

        i = 0
        while len(list(test_ds)) < 1:
            test_ds = generate_inputs_outputs(test_set, past, horizon, 128 - i, 24)
            i += 1

        res = model.predict(test_ds)

        num = sets.replace('.', '_').split('_')[-2]
        _, output = list(test_ds)[0]
        for i in range(res.shape[0]):
            true_y = normalizer_y.inverse_transform(output[i, :].numpy().reshape(-1, 1))
            forecast_y = normalizer_y.inverse_transform(res[i, :].reshape(-1, 1))

            predictions_array = np.append(predictions_array, forecast_y.squeeze())
            true_array = np.append(true_array, true_y.squeeze())

    mse = mean_squared_error(true_array, predictions_array)
    mae = mean_absolute_error(true_array, predictions_array)
    mpe = mean_absolute_percentage_error(true_array, predictions_array)
    metrics = {'Mean Squared Error': mse, 'Mean Absolute Error': mae, 'Mean Absolute Percentage Error': mpe}
    with open(f'results/tests/individual_lstm/{f}/error_metrics_total_128_node.csv', 'w') as error_file:
        w = csv.writer(error_file)
        for key, value in metrics.items():
            w.writerow([key, value])

    fig_arr_forecast = np.split(predictions_array, [horizon*i for i in range(200)])
    fig_arr_true = np.array_split(true_array, [horizon*i for i in range(200)])
    fig_arr_forecast = np.array([x for x in fig_arr_forecast if horizon in x.shape])
    fig_arr_true = np.array([x for x in fig_arr_true if horizon in x.shape])

    rng = np.random.default_rng(0)
    shuffle = rng.permutation(fig_arr_forecast.shape[0])
    fig_arr_forecast = fig_arr_forecast[shuffle, :][:6]
    fig_arr_true = fig_arr_true[shuffle, :][:6]
    fig, axes = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True)
    for j, ax in enumerate(axes.flatten()):
        l1 = ax.plot(fig_arr_true[j, :])
        l2 = ax.plot(fig_arr_forecast[j, :])
        if not (j % 3):
            ax.set(ylabel=r'$PM_{2.5}$')

        if j == 4:
            ax.set(xlabel='Time Steps')

    fig.legend([l1, l2], labels=[r'$y$', r'$\hat{y}$'], loc=7, borderaxespad=0.1)
    fig.savefig(f'results/tests/individual_lstm/{f}/forecast_plots_128_node.png')
    plt.close()

    keras.backend.clear_session()
    print(f'done with {f}')

print('done with 128-128-128-64 node neural network')

# Define hyperparameters and other parameters
epochs = 1000
learning_rate = 1e-2
l1l2 = (0.0, 0.1)
n_features = 468
past = 48
horizon = 24
batch_numbers = [128]*9

opt = keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=100, min_lr=0.0001)

model = Sequential()
model.add(Input(shape=(past, n_features)))
model.add(Bidirectional(LSTM(units=128, return_sequences=True, activity_regularizer=l1_l2(l1l2[0], l1l2[1]))))
model.add(Bidirectional(LSTM(units=128, return_sequences=True, activity_regularizer=l1_l2(l1l2[0], l1l2[1]))))
model.add(Bidirectional(LSTM(units=128, return_sequences=True, activity_regularizer=l1_l2(l1l2[0], l1l2[1]))))
model.add(Bidirectional(LSTM(units=64, kernel_regularizer=l1_l2(l1l2[0], l1l2[1]))))
model.add(Dense(units=horizon))
model.summary()

model.compile(optimizer=opt, loss='mse')

files = os.listdir('dataset/lstm_dataset_splits/individual/')
for idx, f in enumerate(files):
    train_sets = os.listdir(f'dataset/lstm_dataset_splits/individual/{f}/train_sets/')
    dev_sets = os.listdir(f'dataset/lstm_dataset_splits/individual/{f}/dev_sets/')
    test_sets = os.listdir(f'dataset/lstm_dataset_splits/individual/{f}/test_sets/')

    n = np.floor(len(train_sets)/len(dev_sets)).astype(int)
    dev_sets = np.concatenate([dev_sets for i in range(n)]).tolist()
    for i in range(len(train_sets) % len(dev_sets)):
        dev_sets.append(dev_sets[i])

    zip_sets = list(zip_longest(train_sets, dev_sets))
    os.makedirs(f'results/tests/individual_lstm/{f}/plots/', exist_ok=True)
    t0 = time.perf_counter()
    for sets in zip_sets:
        train_set = pd.read_pickle(f'dataset/lstm_dataset_splits/individual/{f}/train_sets/{sets[0]}').to_numpy()
        dev_set = pd.read_pickle(f'dataset/lstm_dataset_splits/individual/{f}/dev_sets/{sets[1]}').to_numpy()

        train_ds = generate_inputs_outputs(train_set, past, horizon, batch_numbers[idx], 24)
        dev_ds = generate_inputs_outputs(dev_set, past, horizon, batch_numbers[idx], 24)

        i = 0
        while len(list(train_ds)) < 1:
            train_ds = generate_inputs_outputs(train_set, past, horizon, batch_numbers[idx] - i, 24)
            i += 1

        i = 0
        while len(list(dev_ds)) < 1:
            dev_ds = generate_inputs_outputs(dev_set, past, horizon, batch_numbers[idx] - i, 24)
            i += 1

        res = model.fit(x=train_ds, validation_data=dev_ds, epochs=epochs, shuffle=False,
                        callbacks=[early_stopping, reduce_lr])

        t1 = time.perf_counter()
        print(f'Time for {early_stopping.stopped_epoch} epochs:', t1 - t0)

    predictions_array = np.array([])
    true_array = np.array([])
    normalizer_y = load(f'dataset/lstm_dataset_splits/individual/{f}/normalizer_y.joblib')
    for sets in test_sets:
        test_set = pd.read_pickle(f'dataset/lstm_dataset_splits/individual/{f}/test_sets/{sets}').to_numpy()
        test_ds = generate_inputs_outputs(test_set, past, horizon, 128, 24)

        i = 0
        while len(list(test_ds)) < 1:
            test_ds = generate_inputs_outputs(test_set, past, horizon, 128 - i, 24)
            i += 1

        res = model.predict(test_ds)

        num = sets.replace('.', '_').split('_')[-2]
        _, output = list(test_ds)[0]
        for i in range(res.shape[0]):
            true_y = normalizer_y.inverse_transform(output[i, :].numpy().reshape(-1, 1))
            forecast_y = normalizer_y.inverse_transform(res[i, :].reshape(-1, 1))

            predictions_array = np.append(predictions_array, forecast_y.squeeze())
            true_array = np.append(true_array, true_y.squeeze())

    mse = mean_squared_error(true_array, predictions_array)
    mae = mean_absolute_error(true_array, predictions_array)
    mpe = mean_absolute_percentage_error(true_array, predictions_array)
    metrics = {'Mean Squared Error': mse, 'Mean Absolute Error': mae, 'Mean Absolute Percentage Error': mpe}
    with open(f'results/tests/individual_lstm/{f}/error_metrics_total_128_node_l2.csv', 'w') as error_file:
        w = csv.writer(error_file)
        for key, value in metrics.items():
            w.writerow([key, value])

    fig_arr_forecast = np.split(predictions_array, [horizon*i for i in range(200)])
    fig_arr_true = np.array_split(true_array, [horizon*i for i in range(200)])
    fig_arr_forecast = np.array([x for x in fig_arr_forecast if horizon in x.shape])
    fig_arr_true = np.array([x for x in fig_arr_true if horizon in x.shape])

    rng = np.random.default_rng(0)
    shuffle = rng.permutation(fig_arr_forecast.shape[0])
    fig_arr_forecast = fig_arr_forecast[shuffle, :][:6]
    fig_arr_true = fig_arr_true[shuffle, :][:6]
    fig, axes = plt.subplots(nrows=2, ncols=3, sharey=True, sharex=True)
    for j, ax in enumerate(axes.flatten()):
        l1 = ax.plot(fig_arr_true[j, :])
        l2 = ax.plot(fig_arr_forecast[j, :])
        if not (j % 3):
            ax.set(ylabel=r'$PM_{2.5}$')

        if j == 4:
            ax.set(xlabel='Time Steps')

    fig.legend([l1, l2], labels=[r'$y$', r'$\hat{y}$'], loc=7, borderaxespad=0.1)
    fig.savefig(f'results/tests/individual_lstm/{f}/forecast_plots_128_node_l2.png')
    plt.close()

    keras.backend.clear_session()
    print(f'done with {f}')

print('done with 128-128-128-64 l2=0.1 node neural network')


