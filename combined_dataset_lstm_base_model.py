import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, LSTM, Dense, Bidirectional

train_set_x = np.load('dataset/lstm_dataset_splits/collective/train_set_x.npy')
train_set_y = np.load('dataset/lstm_dataset_splits/collective/train_set_y.npy')
dev_set_x = np.load('dataset/lstm_dataset_splits/collective/dev_set_x.npy')
dev_set_y = np.load('dataset/lstm_dataset_splits/collective/dev_set_y.npy')
test_set_x = np.load('dataset/lstm_dataset_splits/collective/test_set_x.npy')
test_set_y = np.load('dataset/lstm_dataset_splits/collective/test_set_y.npy')

inputs = Input(shape=(train_set_x.shape[1], train_set_x.shape[2]))
lstm_out = Bidirectional(LSTM(units=296, return_sequences=True))(inputs)
lstm_out = Bidirectional(LSTM(units=148))(inputs)
outputs = Dense(units=24)(lstm_out)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')
model.summary()

# Start hyperparameter tuning with epochs
epochs = [10, 50, 100, 200, 500, 1000, 2000]
batches = 4

train_loss = []
val_loss = []
for e in epochs:
    for i in range(10):
        res = model.fig(x=train_set_x, y=train_set_y, validation_data=(dev_set_x, dev_set_y), epochs=e, batch_size=batches)
        train_loss.append(res.history['loss'])
        val_loss.append(res.history['val_loss'])


print('done')
