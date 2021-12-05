import os
import time
import csv
from joblib import load
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras


# Define hyperparameters and other parameters
epochs = 1000
learning_rate = 1e-2
l1l2 = (0.0, 0.0)
n_features = 18
past = 48
horizon = 24
batch_numbers = 128

opt = keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=50, min_lr=0.0001)

north_list = ['Anchorage', 'Oakland', 'Prague', 'Dhaka', 'Abidjan']
south_list = ['Melbourne', 'Santiago', 'Sao Paulo', 'Thembisa']

os.makedirs('results/tests/non_sequential_splits/transfer_learning/reduced_dim/', exist_ok=True)
networks = os.listdir('dataset/non_sequential_splits/neural_networks/reduced_dim/')
for n_networks in networks:
    for idx, f in enumerate(south_list):
        model = keras.models.load_model(f'dataset/non_sequential_splits/neural_networks/reduced_dim/{n_networks}')

        for layer in model.layers:
            layer.trainable = False

        model.layers[-2].trainable = True
        model.layers[-1].trainable = True

        model.summary()
        model.compile(optimizer=opt, loss='mse')

        train_set_x = np.load(f'dataset/non_sequential_splits/{f}/train_set_x.npy')[:, :, :n_features]
        train_set_y = np.load(f'dataset/non_sequential_splits/{f}/train_set_y.npy')
        dev_set_x = np.load(f'dataset/non_sequential_splits/{f}/dev_set_x.npy')[:, :, :n_features]
        dev_set_y = np.load(f'dataset/non_sequential_splits/{f}/dev_set_y.npy')

        t0 = time.perf_counter()
        res = model.fit(x=train_set_x, y=train_set_y, validation_data=(dev_set_x, dev_set_y), epochs=epochs,
                        shuffle=False, callbacks=[early_stopping, reduce_lr])

        t1 = time.perf_counter()
        print(f'Time for {early_stopping.stopped_epoch} epochs:', t1 - t0)

        print(f'Done with training {f}')

        os.makedirs(f'results/tests/non_sequential_splits/transfer_learning/reduced_dim/{f}/', exist_ok=True)
        normalizer_y = load(f'dataset/non_sequential_splits/{f}/normalizer_y.joblib')
        test_set_x = np.load(f'dataset/non_sequential_splits/{f}/test_set_x.npy')[:, :, :n_features]
        test_set_y = np.load(f'dataset/non_sequential_splits/{f}/test_set_y.npy')

        test_res = model.predict(test_set_x)

        # rescale back from minmax normalizer
        test_res = normalizer_y.inverse_transform(test_res)
        test_set_y = normalizer_y.inverse_transform(test_set_y.squeeze())

        mse = mean_squared_error(test_set_y, test_res)
        mae = mean_absolute_error(test_set_y, test_res)
        mpe = mean_absolute_percentage_error(test_set_y, test_res)
        metrics = {'Mean Squared Error': mse, 'Mean Absolute Error': mae, 'Mean Absolute Percentage Error': mpe}
        with open(f'results/tests/non_sequential_splits/transfer_learning/reduced_dim/{f}/error_metrics_total_individual.csv', 'w') as error_file:
            w = csv.writer(error_file)
            for key, value in metrics.items():
                w.writerow([key, value])

        if test_res.shape[0] < 240:
            i = test_res.shape[0]
        else:
            i = 240

        test_res = np.concatenate(test_res)
        test_set_y = np.concatenate(test_set_y)

        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(test_set_y[:i], label=r'$y$')
        ax[0].plot(test_res[:i], label=r'$\hat{y}$')
        ax[1].plot(np.abs(test_set_y - test_res)[:i])
        ax[0].legend()
        ax[0].set(ylabel=r'$PM_{2.5}$')
        ax[1].set(xlabel='Measurements', ylabel=r'$|y - \hat{y}|$')
        fig.savefig(f'results/tests/non_sequential_splits/transfer_learning/reduced_dim/{f}/forecast_plots_individual.png')
        plt.close()

        keras.backend.clear_session()
