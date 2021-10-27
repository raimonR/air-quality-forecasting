import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, LSTM, Dense, Bidirectional, Dropout

train_set_x = np.load('dataset/lstm_dataset_splits/collective/train_set_x.npy')
train_set_y = np.load('dataset/lstm_dataset_splits/collective/train_set_y.npy')
dev_set_x = np.load('dataset/lstm_dataset_splits/collective/dev_set_x.npy')
dev_set_y = np.load('dataset/lstm_dataset_splits/collective/dev_set_y.npy')
test_set_x = np.load('dataset/lstm_dataset_splits/collective/test_set_x.npy')
test_set_y = np.load('dataset/lstm_dataset_splits/collective/test_set_y.npy')

opt = keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

inputs = Input(shape=(train_set_x.shape[1], train_set_x.shape[2]))
lstm_out = Bidirectional(LSTM(units=128, return_sequences=True))(inputs)
lstm_out_2 = Bidirectional(LSTM(units=64))(lstm_out)
outputs = Dense(units=24)(lstm_out_2)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=opt, loss='mse')
model.summary()

# Start hyperparameter tuning with epochs
epochs = [10, 50, 100, 250, 500]
batches = 32
repeats = 3

train_loss = np.zeros((len(epochs), repeats))
val_loss = np.zeros((len(epochs), repeats))
historical_train_loss = []
historical_val_loss = []
for i, e in enumerate(epochs):
    t0 = time.perf_counter()
    for j in range(repeats):
        res = model.fit(x=train_set_x, y=train_set_y, validation_data=(dev_set_x, dev_set_y),
                        epochs=e, batch_size=batches, callbacks=[callback])
        historical_train_loss.append(res.history['loss'])
        historical_val_loss.append(res.history['val_loss'])
        train_loss[i, j] = res.history['loss'][-1]
        val_loss[i, j] = res.history['val_loss'][-1]

    t1 = time.perf_counter()
    print(f'Total time for {repeats} repeats:', (t1 - t0)/60)
    print(f'Time for {e} epochs:', ((t1 - t0)/repeats)/60)

np.save('epoch_tuning_training_loss', train_loss)
np.save('epoch_tuning_validation_loss', val_loss)

with open('results/tuning/lstm_combined_training.pickle', 'wb') as file:
    pickle.dump(historical_train_loss, file, protocol=-1)

with open('results/tuning/lstm_combined_validation.pickle', 'wb') as file:
    pickle.dump(historical_train_loss, file, protocol=-1)

fig, ax = plt.subplots()
for j in range(repeats):
    for e in epochs:
        ax.plot(e, train_loss[:, j], '.', color='tab:blue')
        ax.plot(e, val_loss[:, j], '.', color='tab:orange')

ax.set(xlabel='Epochs', ylabel='Mean Squared Error')
fig.savefig('epoch_tuning_plot.png')
plt.close()

print('done')
