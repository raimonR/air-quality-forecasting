import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
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
# TODO: tune the regularization more
epochs = 300
batches = 128
learning_rate = 1e-3
repeats = 3
l1l2_weights = [(0, 0), (0.01, 0), (0, 0.01), (0.01, 0.01), (0.001, 0), (0, 0.001), (0.001, 0.001)]
# weights = 1e-1

test_loss = np.zeros((len(l1l2_weights), test_set_y.shape[0], repeats))
for i, (l1, l2) in enumerate(l1l2_weights):
    t0 = time.perf_counter()
    for j in range(repeats):
        opt = keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model = Sequential()
        model.add(Input(shape=(train_set_x.shape[1], train_set_x.shape[2])))
        model.add(Bidirectional(LSTM(units=64, return_sequences=True, bias_regularizer=l1_l2(l1, l2)))) # , recurrent_regularizer=l1_l2(l1, l2))))
        model.add(Bidirectional(LSTM(units=32, bias_regularizer=l1_l2(l1, l2)))) # , recurrent_regularizer=l1_l2(l1, l2))))
        model.add(Dense(units=24))
        model.summary()

        model.compile(optimizer=opt, loss='mse')

        res = model.fit(x=train_set_x, y=train_set_y, validation_data=(dev_set_x, dev_set_y),
                        epochs=epochs, batch_size=batches, callbacks=[callback])

        forecast = model.predict(x=test_set_x)
        test_loss[i, :, j] = ((test_set_y.squeeze() - forecast)**2).mean(axis=1)

        keras.backend.clear_session()

    t1 = time.perf_counter()
    print(f'Total time for {repeats} repeats:', (t1 - t0)/60)
    print('Time for 300 epochs:', ((t1 - t0)/repeats)/60)

test_loss = test_loss.reshape((7, 386*3), order='F')
l1l2_weights = [str(tup) for tup in l1l2_weights]
df = pd.DataFrame(test_loss.T, columns=l1l2_weights).melt(var_name='L1L2 Weights', value_data='MSE')

ax = sb.boxplot(data=df, x='L1L2 Weights', y='MSE', orient='v')
ax.set(title='Bias Regularizer')
fig = ax.get_figure()
fig.savefig('results/tuning/combined_lstm/bias_regularizer.png')
plt.close()

# forecast air quality
# for i in range(repeats):
#     keras.backend.clear_session()
#     model = keras.models.load_model(f'results/tests/combined_lstm/keras_states/version_{j}')
#     test_res = model.predict(test_set_x)
#
#     # metrics
#     mse = mean_squared_error(test_set_y.squeeze(), test_res)
#     mae = mean_absolute_error(test_set_y.squeeze(), test_res)
#     mpe = mean_absolute_percentage_error(test_set_y.squeeze(), test_res)
#
#     error_metrics = {'Mean Squared Error': mse, 'Mean Absolute Error': mae, 'Mean Absolute Percentage Error': mpe}
#     with open('results/tests/combined_lstm/error_metrics.pickle', 'wb') as file:
#         pickle.dump(error_metrics, file, protocol=-1)
#
#     print('Mean Squared Error: ', mse)
#     print('Mean Absolute Error: ', mae)
#     print('Mean Absolute Percentage Error: ', mpe)
#
#     plot_test_y = np.concatenate(test_set_y.squeeze())[:720]
#     plot_forecast_y = np.concatenate(test_res)[:720]
#     fig, ax = plt.subplots(nrows=2, sharex=True)
#     ax[0].plot(plot_test_y, label=r'$y$')
#     ax[0].plot(plot_forecast_y, label=r'$\hat{y}$')
#     ax[1].plot(np.abs(plot_test_y - plot_forecast_y))
#     ax[0].set(ylabel=r'$PM_{2.5}$')
#     ax[1].set(xlabel=r'Time', ylabel=r'$|y-\hat{y}|$')
#     # plt.show()
#     fig.savefig(f'results/tests/combined_lstm/forecast_vs_true_plot_{i}.png')
#     plt.close()

print('done')
