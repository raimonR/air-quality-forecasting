import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Bidirectional
from keras.regularizers import l2


# Define hyperparameters
dr = 0.1
weights = 1e-1
epochs = 300
batches = 128
repeats = 3

files = os.listdir('dataset/lstm_dataset_splits/individual/')
for f in [files[0]]:
    train_set_x = np.load(f'dataset/lstm_dataset_splits/individual/{f}/train_set_x.npy')
    train_set_y = np.load(f'dataset/lstm_dataset_splits/individual/{f}/train_set_y.npy')
    dev_set_x = np.load(f'dataset/lstm_dataset_splits/individual/{f}/dev_set_x.npy')
    dev_set_y = np.load(f'dataset/lstm_dataset_splits/individual/{f}/dev_set_y.npy')
    test_set_x = np.load(f'dataset/lstm_dataset_splits/individual/{f}/test_set_x.npy')
    test_set_y = np.load(f'dataset/lstm_dataset_splits/individual/{f}/test_set_y.npy')

    opt = keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    model = keras.Sequential()
    model.add(Input(shape=(train_set_x.shape[1], train_set_x.shape[2])))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True, dropout=dr, recurrent_dropout=dr,
                                 kernel_regularizer=l2(weights), recurrent_regularizer=l2(weights))))
    model.add(Bidirectional(LSTM(units=32, dropout=dr, recurrent_dropout=dr,
                                 kernel_regularizer=l2(weights), recurrent_regularizer=l2(weights))))
    model.add(Dense(units=24))
    model.summary()

    model.compile(optimizer=opt, loss='mse')

    res = model.fit(x=train_set_x, y=train_set_y, validation_data=(dev_set_x, dev_set_y),
        epochs=epochs, batch_size=batches, callbacks=[callback])


    print('done')
