import time
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
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


# Define hyperparameters
dr = 0.1
weights = 1e-1
epochs = 300
batches = 128
learning_rate = 1e-3
repeats = 5

historical_train_loss = []
historical_val_loss = []
t0 = time.perf_counter()
for j in range(repeats):
    opt = keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # TODO: Decide if i should set recurrent dropout to 0 so that training time speeds up.
    inputs = Input(shape=(train_set_x.shape[1], train_set_x.shape[2]))
    lstm_out = Bidirectional(LSTM(units=64, return_sequences=True, dropout=dr, recurrent_dropout=dr,
                                  kernel_regularizer=l2(weights), recurrent_regularizer=l2(weights)))(inputs)
    lstm_out_2 = Bidirectional(LSTM(units=32, dropout=dr, recurrent_dropout=dr,
                                    kernel_regularizer=l2(weights), recurrent_regularizer=l2(weights)))(lstm_out)
    outputs = Dense(units=24)(lstm_out_2)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=opt, loss='mse')
    model.summary()

    res = model.fit(x=train_set_x, y=train_set_y, validation_data=(dev_set_x, dev_set_y),
        epochs=epochs, batch_size=batches, callbacks=[callback])
    historical_train_loss.append(res.history['loss'])
    historical_val_loss.append(res.history['val_loss'])

    model.save(f'results/tests/combined_lstm/keras_states/version_{j}')
    keras.backend.clear_session()

t1 = time.perf_counter()
print(f'Total time for {repeats} repeats:', (t1 - t0)/60)
print('Time for 300 epochs:', ((t1 - t0)/repeats)/60)

# forecast air quality
for i in range(repeats):
    model = keras.models.load_model(f'results/tests/combined_lstm/keras_states/version_{j}')
    test_res = model.predict(test_set_x)

    # metrics
    mse = mean_squared_error(test_set_y, test_res)
    mae = mean_absolute_error(test_set_y, test_res)
    mpe = mean_absolute_percentage_error(test_set_y, test_res)

    print('Mean Squared Error: ', mse)
    print('Mean Absolute Error: ', mae)
    print('Mean Absolute Percentage Error: ', mpe)

    fig, ax = plt.subplots()
    ax.plot(test_set_y, label=r'$y$')
    ax.plot(test_set_y, label=r'$\hat{y}$')
    ax.set(xlabel=r'$\text{Time}$', ylabel=r'$PM_{2.5}$')
    # plt.show()
    fig.savefig(f'results/tests/combined_lstm/forecast_vs_true_plot_{i}.png')
    plt.close()

with open('results/tests/combined_lstm/lstm_combined_training_learning_rate.pickle', 'wb') as file:
    pickle.dump(historical_train_loss, file, protocol=-1)

with open('results/tuning/combined_lstm/lstm_combined_validation_learning_rate.pickle', 'wb') as file:
    pickle.dump(historical_train_loss, file, protocol=-1)

print('done')
