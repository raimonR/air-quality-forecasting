import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense

files = os.listdir('dataset/lstm_dataset_splits/individual/')
for f in files:
    train_set_x = np.load(f'dataset/lstm_dataset_splits/individual/{f}/train_set_x.npy')
    train_set_y = np.load(f'dataset/lstm_dataset_splits/individual/{f}/train_set_y.npy')
    dev_set_x = np.load(f'dataset/lstm_dataset_splits/individual/{f}/dev_set_x.npy')
    dev_set_y = np.load(f'dataset/lstm_dataset_splits/individual/{f}/dev_set_y.npy')
    test_set_x = np.load(f'dataset/lstm_dataset_splits/individual/{f}/test_set_x.npy')
    test_set_y = np.load(f'dataset/lstm_dataset_splits/individual/{f}/test_set_y.npy')

    inputs = Input(shape=(train_set_x.shape[1], train_set_x.shape[2]))
    lstm_out = Bidirectional(LSTM(units=296, return_sequences=True))(inputs)
    lstm_out = Bidirectional(LSTM(units=148))(lstm_out)
    outputs = Dense(units=24)(lstm_out)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    model.summary()


