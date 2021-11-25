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


north_list = ['Anchorage', 'Oakland', 'Prague', 'Dhaka', 'Abidjan']
south_list = ['Melbourne', 'Santiago', 'Sao Paulo', 'Thembisa']

for n in range(1, len(north_list) + 1):
    # Define hyperparameters and other parameters
    epochs = 750
    learning_rate = 1e-2
    l1l2 = (0.0, 0.0)
    n_features = 468
    past = 48
    horizon = 24
    batch_numbers = [128]*9

    opt = keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=60, min_lr=0.0001)

    model = Sequential()
    model.add(Input(shape=(past, n_features)))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True, activity_regularizer=l1_l2(l1l2[0], l1l2[1]))))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True, activity_regularizer=l1_l2(l1l2[0], l1l2[1]))))
    model.add(Bidirectional(LSTM(units=64, kernel_regularizer=l1_l2(l1l2[0], l1l2[1]))))
    model.add(Dense(units=horizon))
    model.summary()

    model.compile(optimizer=opt, loss='mse')
    for loc in north_list[:n]:
        print(loc)
