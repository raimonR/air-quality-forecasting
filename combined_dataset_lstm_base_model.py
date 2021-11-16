import time
from joblib import load
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Input, LSTM, Dense, Bidirectional
from keras.regularizers import l1_l2


train_set_x = np.load('dataset/lstm_dataset_splits/collective/train_set_x.npy')
train_set_y = np.load('dataset/lstm_dataset_splits/collective/train_set_y.npy')
dev_set_x = np.load('dataset/lstm_dataset_splits/collective/dev_set_x.npy')
dev_set_y = np.load('dataset/lstm_dataset_splits/collective/dev_set_y.npy')
test_set_x = np.load('dataset/lstm_dataset_splits/collective/test_set_x.npy')
test_set_y = np.load('dataset/lstm_dataset_splits/collective/test_set_y.npy')

# Define hyperparameters
epochs = 500
batches = 128
learning_rate = 1e-3
l1l2 = (0.1, 0.1)

t0 = time.perf_counter()
opt = keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model = Sequential()
model.add(Input(shape=(train_set_x.shape[1], train_set_x.shape[2])))
model.add(Bidirectional(LSTM(units=64, return_sequences=True, activity_regularizer=l1_l2(l1l2[0], l1l2[1]))))
model.add(Bidirectional(LSTM(units=32, activity_regularizer=l1_l2(l1l2[0], l1l2[1]))))
model.add(Dense(units=24))
model.summary()

model.compile(optimizer=opt, loss='mse')

res = model.fit(x=train_set_x, y=train_set_y, validation_data=(dev_set_x, dev_set_y), shuffle=False,
                epochs=epochs, batch_size=batches, callbacks=[callback])

t1 = time.perf_counter()
print(f'Time for {callback.stopped_epoch} epochs:', t1 - t0)

model.save(f'results/tests/combined_lstm/keras_states/combined_model')

# forecast air quality
normalizer = load(f'dataset/lstm_dataset_splits/collective/normalizer_y.joblib')
test_res = model.predict(test_set_x)

# rescale back from minmax normalizer
test_res = normalizer.inverse_transform(test_res)
test_set_y = normalizer.inverse_transform(test_set_y.squeeze())

# metrics
mse = mean_squared_error(test_set_y, test_res)
mae = mean_absolute_error(test_set_y, test_res)
mpe = mean_absolute_percentage_error(test_set_y, test_res)

error_metrics = {'Mean Squared Error': mse, 'Mean Absolute Error': mae, 'Mean Absolute Percentage Error': mpe}
with open('results/tests/combined_lstm/error_metrics.pickle', 'wb') as file:
    pickle.dump(error_metrics, file, protocol=-1)

print('Mean Squared Error: ', mse)
print('Mean Absolute Error: ', mae)
print('Mean Absolute Percentage Error: ', mpe)

plot_test_y = np.concatenate(test_set_y)
plot_forecast_y = np.concatenate(test_res)
fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(plot_test_y, label=r'$y$')
ax[0].plot(plot_forecast_y, label=r'$\hat{y}$')
ax[1].plot(np.abs(plot_test_y - plot_forecast_y))
ax[0].set(ylabel=r'Normalized $PM_{2.5}$')
ax[1].set(xlabel=r'Measurements', ylabel=r'$|y-\hat{y}|$')
ax[0].legend(fontsize=5)
# plt.show()
fig.savefig(f'results/tests/combined_lstm/forecast_vs_true_plot.png')
plt.close()

print('done with training, validation, and testing')

set_x = np.load('dataset/lstm_dataset_splits/collective/set_x_thembisa.npy')
set_y = np.load('dataset/lstm_dataset_splits/collective/set_y_thembisa.npy')
normalizer_y = load('dataset/lstm_dataset_splits/collective/normalizer_y_thembisa.joblib')

test_error = model.evaluate(x=set_x, y=set_y)
print(test_error)
forecast = model.predict(set_x)
true_y = np.concatenate(normalizer_y.inverse_transform(set_y.squeeze()))
forecast = np.concatenate(normalizer_y.inverse_transform(forecast.squeeze()))

fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(true_y, label=r'$y$')
ax[0].plot(forecast, label=r'$\hat{y}$')
ax[1].plot(np.abs(true_y - forecast))
ax[0].set(ylabel=r'Normalized $PM_{2.5}$')
ax[1].set(xlabel=r'Measurements', ylabel=r'$|y-\hat{y}|$')
ax[0].legend(fontsize=5)
fig.savefig(f'results/tests/combined_lstm/thembisa_forecast_plot.png')

print('done')
