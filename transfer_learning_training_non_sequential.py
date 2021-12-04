import os
import time
from itertools import zip_longest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Bidirectional
from keras.regularizers import l1_l2


# Define hyperparameters and other parameters
epochs = 1000
learning_rate = 1e-2
l1l2 = (0.0, 0.0)
n_features = 468
past = 48
horizon = 24
batch_numbers = 128

north_list = ['Anchorage', 'Oakland', 'Prague', 'Dhaka', 'Abidjan']
os.makedirs('dataset/non_sequential_splits/neural_networks/', exist_ok=True)
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
        train_set_x = np.load(f'dataset/non_sequential_splits/{loc}/train_set_x.npy')
        train_set_y = np.load(f'dataset/non_sequential_splits/{loc}/train_set_y.npy')
        dev_set_x = np.load(f'dataset/non_sequential_splits/{loc}/dev_set_x.npy')
        dev_set_y = np.load(f'dataset/non_sequential_splits/{loc}/dev_set_y.npy')

        t0 = time.perf_counter()
        res = model.fit(x=train_set_x, y=train_set_y, validation_data=(dev_set_x, dev_set_y), epochs=epochs,
                        shuffle=False, callbacks=[early_stopping, reduce_lr])

        t1 = time.perf_counter()
        print(f'Time for {early_stopping.stopped_epoch} epochs:', t1 - t0)

    model.save(f'dataset/non_sequential_splits/neural_networks/full_dim/tl_n{n_iter}')

    keras.backend.clear_session()
