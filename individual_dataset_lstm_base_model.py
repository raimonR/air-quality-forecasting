import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow import data
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Bidirectional
from keras.regularizers import l2
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow import data, autograph

autograph.set_verbosity(0)


def dataset_generator(inputs, output_dims, input_length, output_length, batch_size):
    dataset = timeseries_dataset_from_array(inputs, None, input_length + output_length, batch_size=batch_size)

    def split_inputs(x):
        return x[:, :input_length, :], x[:, input_length:, output_dims]

    dataset = dataset.map(split_inputs)

    return dataset


files = os.listdir('dataset/lstm_dataset_splits/individual/')
for f in [files[0]]:
    train_set = np.load(f'dataset/lstm_dataset_splits/individual/{f}/train_set.npy')
    dev_set = np.load(f'dataset/lstm_dataset_splits/individual/{f}/dev_set.npy')
    test_set = np.load(f'dataset/lstm_dataset_splits/individual/{f}/test_set.npy')

    train_set = dataset_generator(train_set, slice(0, 1), input_length=24, output_length=24, batch_size=128)
    dev_set = dataset_generator(dev_set, slice(0, 1), input_length=24, output_length=24, batch_size=128)
    test_set = dataset_generator(test_set, slice(0, 1), input_length=24, output_length=24, batch_size=128)

    # Define hyperparameters
    dr = 0.1
    weights = 1e-1
    epochs = 300
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
