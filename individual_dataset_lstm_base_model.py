import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow import data
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Bidirectional
from keras.regularizers import l2


files = os.listdir('dataset/lstm_dataset_splits/individual/')
for f in [files[0]]:
    train_set = data.experimental.load(f'dataset/lstm_dataset_splits/individual/{f}/train_set')
    dev_set = data.experimental.load(f'dataset/lstm_dataset_splits/individual/{f}/dev_set')
    test_set = data.experimental.load(f'dataset/lstm_dataset_splits/individual/{f}/test_set')

    # Define hyperparameters
    dr = 0.1
    weights = 1e-1
    epochs = 300
    # TODO: determine the best number of batches since the datasets are significantly smaller
    batches = 128
    learning_rate = 1e-3
    repeats = 3

    opt = keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model = Sequential()
    model.add(Input(shape=(train_set.shape[1], train_set.shape[2])))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True, dropout=dr, recurrent_dropout=dr,
                                 kernel_regularizer=l2(weights), recurrent_regularizer=l2(weights))))
    model.add(Bidirectional(LSTM(units=32, dropout=dr, recurrent_dropout=dr,
                                 kernel_regularizer=l2(weights), recurrent_regularizer=l2(weights))))
    model.add(Dense(units=24))
    model.summary()

    model.compile(optimizer=opt, loss='mse')

    res = model.fit(x=train_set, validation_data=dev_set,
                    epochs=epochs, batch_size=batches, callbacks=[callback])


print('done')
