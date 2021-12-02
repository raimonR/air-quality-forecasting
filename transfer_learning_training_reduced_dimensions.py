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
n_features = 18
past = 48
horizon = 24
batch_numbers = 128

north_list = ['Anchorage', 'Oakland', 'Prague', 'Dhaka', 'Abidjan']
os.makedirs('dataset/transfer_learning/neural_networks/reduced_dim/', exist_ok=True)
for n_iter in range(1, len(north_list) + 1):
    print(f'Starting {north_list[:n_iter]}')
    opt = keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=50, min_lr=0.0001)

    model = Sequential()
    model.add(Input(shape=(past, n_features)))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True, activity_regularizer=l1_l2(l1l2[0], l1l2[1]))))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True, activity_regularizer=l1_l2(l1l2[0], l1l2[1]))))
    model.add(Bidirectional(LSTM(units=64, kernel_regularizer=l1_l2(l1l2[0], l1l2[1])), name='transfer_learning'))
    model.add(Dense(units=horizon, name='output'))
    model.summary()

    model.compile(optimizer=opt, loss='mse')

    for loc in north_list[:n_iter]:
        train_sets = os.listdir(f'dataset/transfer_learning/{loc}/train_sets/')
        dev_sets = os.listdir(f'dataset/transfer_learning/{loc}/dev_sets/')

        n = np.floor(len(train_sets)/len(dev_sets)).astype(int)
        dev_sets = np.concatenate([dev_sets for i in range(n)]).tolist()
        for i in range(len(train_sets)%len(dev_sets)):
            dev_sets.append(dev_sets[i])

        zip_sets = list(zip_longest(train_sets, dev_sets))
        t0 = time.perf_counter()
        for sets in zip_sets:
            train_set = pd.read_pickle(f'dataset/transfer_learning/{loc}/train_sets/{sets[0]}').to_numpy()[:, :18]
            dev_set = pd.read_pickle(f'dataset/transfer_learning/{loc}/dev_sets/{sets[1]}').to_numpy()[:, :18]

            train_ds = generate_inputs_outputs(train_set, past, horizon, batch_numbers, 1)
            dev_ds = generate_inputs_outputs(dev_set, past, horizon, batch_numbers, 1)

            i = 0
            while len(list(train_ds)) < 1:
                train_ds = generate_inputs_outputs(train_set, past, horizon, batch_numbers - i, 1)
                i += 1

            i = 0
            while len(list(dev_ds)) < 1:
                dev_ds = generate_inputs_outputs(dev_set, past, horizon, batch_numbers - i, 1)
                i += 1

            res = model.fit(x=train_ds, validation_data=dev_ds, epochs=epochs, shuffle=False,
                            callbacks=[early_stopping, reduce_lr])

            t1 = time.perf_counter()
            print(f'Time for {early_stopping.stopped_epoch} epochs:', t1 - t0)

    model.save(f'dataset/transfer_learning/neural_networks/reduced_dim/tl_n{n_iter}')

    keras.backend.clear_session()
