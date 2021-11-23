import os
import csv
import time
from joblib import load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Input, LSTM, Dense, Bidirectional
from keras.regularizers import l1_l2


os.makedirs('results/tests/combined_lstm/', exist_ok=True)
train_set_x = np.load('dataset/lstm_dataset_splits/collective/train_set_x.npy')
train_set_y = np.load('dataset/lstm_dataset_splits/collective/train_set_y.npy')
dev_set_x = np.load('dataset/lstm_dataset_splits/collective/dev_set_x.npy')
dev_set_y = np.load('dataset/lstm_dataset_splits/collective/dev_set_y.npy')
test_set_x = np.load('dataset/lstm_dataset_splits/collective/test_set_x.npy')
test_set_y = np.load('dataset/lstm_dataset_splits/collective/test_set_y.npy')

# Define hyperparameters
epochs = 1000
batches = 128
learning_rate = 1e-2
l1l2 = (0.0, 0.0)
horizon = 24

t0 = time.perf_counter()
opt = keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=100, min_lr=0.0001)

model = Sequential()
model.add(Input(shape=(train_set_x.shape[1], train_set_x.shape[2])))
model.add(Bidirectional(LSTM(units=128, return_sequences=True, activity_regularizer=l1_l2(l1l2[0], l1l2[1]))))
model.add(Bidirectional(LSTM(units=128, return_sequences=True, activity_regularizer=l1_l2(l1l2[0], l1l2[1]))))
model.add(Bidirectional(LSTM(units=128, return_sequences=True, activity_regularizer=l1_l2(l1l2[0], l1l2[1]))))
model.add(Bidirectional(LSTM(units=64, kernel_regularizer=l1_l2(l1l2[0], l1l2[1]))))
model.add(Dense(units=horizon))
model.summary()

model.compile(optimizer=opt, loss='mse')

res = model.fit(x=train_set_x, y=train_set_y, validation_data=(dev_set_x, dev_set_y), shuffle=False,
                epochs=epochs, batch_size=batches, callbacks=[early_stopping, reduce_lr])

t1 = time.perf_counter()
print(f'Time for {early_stopping.stopped_epoch} epochs:', t1 - t0)

model.save(f'results/tests/combined_lstm/combined_model')

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

metrics = {'Mean Squared Error': mse, 'Mean Absolute Error': mae, 'Mean Absolute Percentage Error': mpe}
with open('results/tests/combined_lstm/error_metrics.csv', 'w') as error_file:
    w = csv.writer(error_file)
    for key, value in metrics.items():
        w.writerow([key, value])

print('Mean Squared Error: ', mse)
print('Mean Absolute Error: ', mae)
print('Mean Absolute Percentage Error: ', mpe)

plot_test_y = np.concatenate(test_set_y)
plot_forecast_y = np.concatenate(test_res)
index = np.random.randint(720, plot_forecast_y.shape[0])

fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(plot_test_y[(index - 720):index], label=r'$y$')
ax[0].plot(plot_forecast_y[(index - 720):index], label=r'$\hat{y}$')
ax[1].plot(np.abs(plot_test_y - plot_forecast_y)[(index - 720):index])
ax[0].set(ylabel=r'$PM_{2.5}$')
ax[1].set(xlabel=r'Measurements', ylabel=r'$|y-\hat{y}|$')
ax[0].legend(fontsize=5)
# plt.show()
fig.savefig(f'results/tests/combined_lstm/forecast_vs_true_plot.png')
plt.close()

# Thembisa test
set_x = np.load('dataset/lstm_dataset_splits/collective/set_x_thembisa.npy')
set_y = np.load('dataset/lstm_dataset_splits/collective/set_y_thembisa.npy')
normalizer_y = load('dataset/lstm_dataset_splits/collective/normalizer_y_thembisa.joblib')

forecast = model.predict(set_x)
true_y = np.concatenate(normalizer_y.inverse_transform(set_y.squeeze()))
forecast = np.concatenate(normalizer_y.inverse_transform(forecast.squeeze()))

mse = mean_squared_error(true_y, forecast)
mae = mean_absolute_error(true_y, forecast)
mpe = mean_absolute_percentage_error(true_y, forecast)

error_metrics = {'Mean Squared Error': mse, 'Mean Absolute Error': mae, 'Mean Absolute Percentage Error': mpe}
with open('results/tests/combined_lstm/error_metrics_thembisa.csv', 'w') as error_file:
    w = csv.writer(error_file)
    for key, value in error_metrics.items():
        w.writerow([key, value])

print('Thembisa')
print('Mean Squared Error: ', mse)
print('Mean Absolute Error: ', mae)
print('Mean Absolute Percentage Error: ', mpe)


index = np.random.randint(720, forecast.shape[0])
fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(true_y[(index - 720):index], label=r'$y$')
ax[0].plot(forecast[(index - 720):index], label=r'$\hat{y}$')
ax[1].plot(np.abs(true_y - forecast)[(index - 720):index])
ax[0].set(ylabel=r'$PM_{2.5}$')
ax[1].set(xlabel=r'Measurements', ylabel=r'$|y-\hat{y}|$')
ax[0].legend(fontsize=5)
fig.savefig(f'results/tests/combined_lstm/thembisa_forecast_plot.png')
plt.close()

print('done')
