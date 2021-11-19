import os
import time
import pickle
import numpy as np
import pandas as pd
import csv
from joblib import load
from itertools import zip_longest
import matplotlib.pyplot as plt
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


# Define Parameters
epochs = 500
learning_rate = 1e-2
l1l2 = (0.0, 0.0)
n_features = 468
past = 48
horizon = 24
batch_numbers = [128]*9

opt = keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=25, min_lr=0.001)

model = Sequential()
model.add(Input(shape=(past, n_features)))
model.add(Bidirectional(LSTM(units=64, return_sequences=True, activity_regularizer=l1_l2(l1l2[0], l1l2[1]))))
model.add(Bidirectional(LSTM(units=32, kernel_regularizer=l1_l2(l1l2[0], l1l2[1]))))
model.add(Dense(units=horizon))
model.summary()

model.compile(optimizer=opt, loss='mse')

os.makedirs('results/tests/multifit/', exist_ok=True)
files = os.listdir('dataset/transfer_learning/')
for idx, f in enumerate(files):
    train_sets = os.listdir(f'dataset/transfer_learning/{f}/train_sets/')
    dev_sets = os.listdir(f'dataset/transfer_learning/{f}/dev_sets/')
    test_sets = os.listdir(f'dataset/transfer_learning/{f}/test_sets/')

    n = np.floor(len(train_sets)/len(dev_sets)).astype(int)
    dev_sets = np.concatenate([dev_sets for i in range(n)]).tolist()
    for i in range(len(train_sets) % len(dev_sets)):
        dev_sets.append(dev_sets[i])

    zip_sets = list(zip_longest(train_sets, dev_sets))
    t0 = time.perf_counter()
    for sets in zip_sets:
        train_set = pd.read_pickle(f'dataset/transfer_learning/{f}/train_sets/{sets[0]}').to_numpy()
        dev_set = pd.read_pickle(f'dataset/transfer_learning/{f}/dev_sets/{sets[1]}').to_numpy()

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

    print(f'Done with {f}')

for f in files:
    os.makedirs(f'results/tests/multifit/{f}/plots/', exist_ok=True)
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
            true_y = normalizer_y.inverse_transform(output[i, :])
            forecast_y = normalizer_y.inverse_transform(res[i, :])

            predictions_array = np.append(predictions_array, forecast_y)
            true_array = np.append(true_array, true_y)

            fig, ax = plt.subplots(nrows=2, sharex=True)
            ax[0].plot(true_y, label=r'$y$')
            ax[0].plot(forecast_y, label=r'$\hat{y}$')
            ax[1].plot(np.abs(true_y - forecast_y))
            ax[0].set(ylabel=r'$PM_{2.5}$')
            ax[1].set(ylabel=r'$|y-\hat{y}|$', xlabel='Time Steps')
            ax[0].legend()
            fig.savefig(f'results/tests/multifit/{f}/plots/forecast_plots_{num}_{i}.png')
            plt.close()

            # metrics
            mse = mean_squared_error(true_y, forecast_y)
            mae = mean_absolute_error(true_y, forecast_y)
            mpe = mean_absolute_percentage_error(true_y, forecast_y)
            metrics = {'Mean Squared Error': mse, 'Mean Absolute Error': mae, 'Mean Absolute Percentage Error': mpe}
            with open(f'results/tests/multifit/{f}/error_metrics_{num}_{i}.csv', 'w') as error_file:
                w = csv.writer(error_file)
                for key, value in metrics.items():
                    w.writerow([key, value])

    mse = mean_squared_error(true_array, predictions_array)
    mae = mean_absolute_error(true_array, predictions_array)
    mpe = mean_absolute_percentage_error(true_array, predictions_array)
    metrics = {'Mean Squared Error': mse, 'Mean Absolute Error': mae, 'Mean Absolute Percentage Error': mpe}
    with open(f'results/tests/multifit/{f}/error_metrics_total.csv', 'w') as error_file:
        w = csv.writer(error_file)
        for key, value in metrics.items():
            w.writerow([key, value])

print('done')
