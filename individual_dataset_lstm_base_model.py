import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Bidirectional
from keras.regularizers import l1_l2
from tensorflow import autograph

autograph.set_verbosity(0)


def generate_inputs_outputs(data, n_past, n_horizon, batch_size, shift):
    def make_batch(x):
        return x.batch(length)

    def make_split(x):
        return x[:-n_horizon], x[-n_horizon:, 0]

    length = n_past + n_horizon
    ds = tf.data.Dataset.from_tensor_slices(data)

    ds = ds.window(length, shift=shift, drop_remainder=True)
    ds = ds.flat_map(make_batch)

    ds = ds.map(make_split)

    ds = ds.batch(batch_size)
    return ds


batch_numbers = [128, 128, 128, 128, 128, 128]
files = os.listdir('dataset/lstm_dataset_splits/individual/')
for idx, f in enumerate(files):
    train_set = np.load(f'dataset/lstm_dataset_splits/individual/{f}/train_set.npy')
    dev_set = np.load(f'dataset/lstm_dataset_splits/individual/{f}/dev_set.npy')
    test_set = np.load(f'dataset/lstm_dataset_splits/individual/{f}/test_set.npy')
    
    n_features = train_set.shape[1]
    past = 24
    horizon = 24

    train_ds = generate_inputs_outputs(train_set, past, horizon, batch_numbers[idx], 1)
    dev_ds = generate_inputs_outputs(dev_set, past, horizon, batch_numbers[idx], 1)
    test_ds = generate_inputs_outputs(test_set, past, horizon, 1, 24)

    # Define hyperparameters
    epochs = 500
    learning_rate = 1e-1
    repeats = 2
    l1l2 = (0.1, 0.1)
    
    os.makedirs(f'results/tests/individual_lstm/{f}/keras_states/', exist_ok=True)
    t0 = time.perf_counter()
    for i in range(repeats):
        opt = keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

        model = Sequential()
        model.add(Input(shape=(past, n_features)))
        # model.add(Bidirectional(LSTM(units=32, return_sequences=True, kernel_regularizer=l1_l2(l1l2[0], l1l2[1]))))
        model.add(Bidirectional(LSTM(units=32, kernel_regularizer=l1_l2(l1l2[0], l1l2[1]))))
        model.add(Dense(units=horizon))
        model.summary()

        model.compile(optimizer=opt, loss='mse')
    
        res = model.fit(x=train_ds, validation_data=dev_ds, epochs=epochs, shuffle=False,
                        callbacks=[early_stopping, reduce_lr])
                        
        t1 = time.perf_counter()
        print(f'Time for {early_stopping.stopped_epoch} epochs:', t1 - t0)
        
        forecast = model.evaluate(test_ds, return_dict=True)
        print(forecast)

        predictions = np.array([])
        true_values = np.array([])
        for batch in test_ds.as_numpy_iterator():
            input_tensor, output_tensor = batch
            res = model.predict_on_batch(input_tensor)
            predictions = np.append(predictions, res.squeeze())
            true_values = np.append(true_values, output_tensor.squeeze())

        # metrics
        mse = mean_squared_error(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        mpe = mean_absolute_percentage_error(true_values, predictions)
    
        error_metrics = {'Mean Squared Error': mse, 'Mean Absolute Error': mae, 'Mean Absolute Percentage Error': mpe}
        with open(f'results/tests/individual_lstm/{f}/error_metrics.pickle', 'wb') as file:
            pickle.dump(error_metrics, file, protocol=-1)

        print('Mean Squared Error: ', mse)
        print('Mean Absolute Error: ', mae)
        print('Mean Absolute Percentage Error: ', mpe)

        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(true_values, label=r'$y$')
        ax[0].plot(predictions, label=r'$\hat{y}$')
        ax[1].plot(np.abs(true_values - predictions))
        ax[0].set(ylabel=r'Normalized $PM_{2.5}$')
        ax[1].set(xlabel=r'Measurements', ylabel=r'$|y-\hat{y}|$')
        # plt.show()
        fig.savefig(f'results/tests/individual_lstm/{f}/forecast_vs_true_plot_{i}.png')
        plt.close()

        keras.backend.clear_session()
    
    print(f'done with {f}')

print('done')
