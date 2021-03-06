import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, LSTM, Dense, Bidirectional
from keras.regularizers import l2

train_set_x = np.load('dataset/lstm_dataset_splits/collective/train_set_x.npy')
train_set_y = np.load('dataset/lstm_dataset_splits/collective/train_set_y.npy')
dev_set_x = np.load('dataset/lstm_dataset_splits/collective/dev_set_x.npy')
dev_set_y = np.load('dataset/lstm_dataset_splits/collective/dev_set_y.npy')
test_set_x = np.load('dataset/lstm_dataset_splits/collective/test_set_x.npy')
test_set_y = np.load('dataset/lstm_dataset_splits/collective/test_set_y.npy')

opt = keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

inputs = Input(shape=(train_set_x.shape[1], train_set_x.shape[2]))
lstm_out = Bidirectional(LSTM(units=64, return_sequences=True, dropout=0.25, kernel_regularizer=l2(1e-4)))(inputs)
lstm_out_2 = Bidirectional(LSTM(units=32, dropout=0.25, kernel_regularizer=l2(1e-4)))(lstm_out)
# lstm_out_2 = Bidirectional(LSTM(units=1))(inputs)
outputs = Dense(units=24)(lstm_out_2)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=opt, loss='mse')
model.summary()

# Start hyperparameter tuning with epochs
epochs = 300
batches = [8, 32, 64, 128, 256]
repeats = 3

train_loss = np.zeros((len(batches), repeats))
val_loss = np.zeros((len(batches), repeats))
historical_train_loss = []
historical_val_loss = []
for i, b in enumerate(batches):
    t0 = time.perf_counter()
    for j in range(repeats):
        res = model.fit(x=train_set_x, y=train_set_y, validation_data=(dev_set_x, dev_set_y),
                        epochs=epochs, batch_size=b, callbacks=[callback])
        historical_train_loss.append(res.history['loss'])
        historical_val_loss.append(res.history['val_loss'])
        train_loss[i, j] = res.history['loss'][-1]
        val_loss[i, j] = res.history['val_loss'][-1]

    t1 = time.perf_counter()
    print(f'Total time for {repeats} repeats:', (t1 - t0)/60)
    print(f'Time for {b} epochs:', ((t1 - t0)/repeats)/60)

np.save('results/tuning/epoch_tuning_training_loss_batches', train_loss)
np.save('results/tuning/epoch_tuning_validation_loss_batches', val_loss)

with open('results/tuning/lstm_combined_training_batches.pickle', 'wb') as file:
    pickle.dump(historical_train_loss, file, protocol=-1)

with open('results/tuning/lstm_combined_validation_batches.pickle', 'wb') as file:
    pickle.dump(historical_train_loss, file, protocol=-1)

print('done')
