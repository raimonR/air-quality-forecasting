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
epochs = [1, 10, 50, 100, 250, 500, 1000]
batches = 4

train_loss = np.zeros((len(epochs), 10))
val_loss = np.zeros((len(epochs), 10))
for i, e in enumerate(epochs):
    for j in range(10):
        res = model.fit(x=train_set_x, y=train_set_y, validation_data=(dev_set_x, dev_set_y), epochs=e, batch_size=batches)
        train_loss[i, j] = res.history['loss'][-1]
        val_loss[i, j] = res.history['val_loss'][-1]

np.save('epoch_tuning_training_loss', train_loss)
np.save('epoch_tuning_validation_loss', val_loss)

fig, ax = plt.subplots()
for i, e in enumerate(epochs):
    for j in range(10):
        ax.semilogx(e, train_loss[i, j], '.', color='tab:blue')
        ax.semilogx(e, val_loss[i, j], '.', color='tab:orange')

ax.set(xlabel='Epochs', ylabel='Mean Squared Error')
fig.savefig('epoch_tuning_plot.png')
plt.close()


print('done')
