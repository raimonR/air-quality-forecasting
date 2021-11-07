import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Bidirectional
from keras.regularizers import l1_l2
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow import autograph

autograph.set_verbosity(0)


def dataset_generator(inputs, output_dims, input_length, output_length, batch_size):
    dataset = timeseries_dataset_from_array(inputs, None, input_length + output_length, batch_size=batch_size)

    def split_inputs(x):
        return x[:, :input_length, :], x[:, input_length:, output_dims]

    dataset = dataset.map(split_inputs)

    return dataset

files = os.listdir('dataset/lstm_dataset_splits/individual/')
for f in files:
    train_set = np.load(f'dataset/lstm_dataset_splits/individual/{f}/train_set.npy')
    dev_set = np.load(f'dataset/lstm_dataset_splits/individual/{f}/dev_set.npy')
    test_set = np.load(f'dataset/lstm_dataset_splits/individual/{f}/test_set.npy')
    
    # n_features = train_set.shape[2]
    print(train_set.shape)
    horizon = 24
    past = 24

    train_set_ds = dataset_generator(train_set, slice(0, 1), input_length=past, output_length=horizon, batch_size=128)
    dev_set_ds = dataset_generator(dev_set, slice(0, 1), input_length=past, output_length=horizon, batch_size=128)
    test_set_ds = dataset_generator(test_set, slice(0, 1), input_length=past, output_length=horizon, batch_size=128)

    # Define hyperparameters
    epochs = 500
    batches = 128
    learning_rate = 1e-1
    repeats = 3
    l1l2 = (0.1, 0.1)
    
    os.makedirs(f'results/tests/individual_lstm/{f}/keras_states/', exist_ok=True)
    t0 = time.perf_counter()
    for i in range(repeats):
        opt = keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)
    
        model = Sequential()
        model.add(Input(shape=(past, 468)))
        model.add(Bidirectional(LSTM(units=64, return_sequences=True, kernel_regularizer=l1_l2(l1l2[0], l1l2[1]))))
        model.add(Bidirectional(LSTM(units=32, kernel_regularizer=l1_l2(l1l2[0], l1l2[1]))))
        model.add(Dense(units=horizon))
        model.summary()
    
        model.compile(optimizer=opt, loss='mse')
    
        res = model.fit(x=train_set_ds, validation_data=dev_set_ds, epochs=epochs, shuffle=False, batch_size=batches,
                        callbacks=[early_stopping, reduce_lr])
                        
        t1 = time.perf_counter()
        print(f'Time for {early_stopping.stopped_epoch} epochs:', t1 - t0)
        
        forecast = model.predict(test_set_ds)
    
        # metrics
        mse = mean_squared_error(test_set_ds.squeeze(), forecast)
        mae = mean_absolute_error(test_set_ds.squeeze(), forecast)
        mpe = mean_absolute_percentage_error(test_set_ds.squeeze(), forecast)
    
        error_metrics = {'Mean Squared Error': mse, 'Mean Absolute Error': mae, 'Mean Absolute Percentage Error': mpe}
        with open('results/tests/individual_lstm/{f}/error_metrics.pickle', 'wb') as file:
            pickle.dump(error_metrics, file, protocol=-1)
    
        print('Mean Squared Error: ', mse)
        print('Mean Absolute Error: ', mae)
        print('Mean Absolute Percentage Error: ', mpe)
    
        plot_test_y = np.concatenate(test_set_y.squeeze())[:720]
        plot_forecast_y = np.concatenate(forecast)[:720]
        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(plot_test_y, label=r'$y$')
        ax[0].plot(plot_forecast_y, label=r'$\hat{y}$')
        ax[1].plot(np.abs(plot_test_y - plot_forecast_y))
        ax[0].set(ylabel=r'Normalized $PM_{2.5}$')
        ax[1].set(xlabel=r'Measurements', ylabel=r'$|y-\hat{y}|$')
        # plt.show()
        fig.savefig(f'results/tests/individual_lstm/{f}/forecast_vs_true_plot_{i}.png')
        plt.close()
        
        model.save(f'results/tests/individual_lstm/{f}/keras_states/version_{j}')
        keras.backend.clear_session()
    
    print(f'done with {f}')

print('done')
