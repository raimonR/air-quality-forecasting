import os
import time
import csv
from joblib import load
from itertools import zip_longest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from tensorflow import autograph

autograph.set_verbosity(0)


@tf.autograph.experimental.do_not_convert
def generate_inputs_outputs(data, n_past, n_horizon, batch_num, shift):
    def make_batch(x):
        return x.batch(length, drop_remainder=True)

    def make_split(x):
        return x[:-n_horizon], x[-n_horizon:, 0]

    length = n_past + n_horizon
    ds = tf.data.Dataset.from_tensor_slices(data)

    ds = ds.window(length, shift=shift, drop_remainder=True)
    ds = ds.flat_map(make_batch)

    ds = ds.map(make_split)

    ds = ds.batch(batch_num, drop_remainder=True)
    return ds


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

os.makedirs('results/tests/transfer_learning/', exist_ok=True)
networks = os.listdir('dataset/transfer_learning/neural_networks/reduced_dim/')
for n_networks in networks:
    model = keras.models.load_model(f'dataset/transfer_learning/neural_networks/reduced_dim/{n_networks}')

    for layer in model.layers:
        layer.trainable = False

    model.layers[-2].trainable = True
    model.layers[-1].trainable = True

    model.summary()
    model.compile(optimizer=opt, loss='mse')

    for idx, f in enumerate(south_list):
        train_sets = os.listdir(f'dataset/transfer_learning/{f}/train_sets/')
        dev_sets = os.listdir(f'dataset/transfer_learning/{f}/dev_sets/')

        n = np.floor(len(train_sets)/len(dev_sets)).astype(int)
        dev_sets = np.concatenate([dev_sets for i in range(n)]).tolist()
        for i in range(len(train_sets) % len(dev_sets)):
            dev_sets.append(dev_sets[i])

        zip_sets = list(zip_longest(train_sets, dev_sets))
        t0 = time.perf_counter()
        for sets in zip_sets:
            train_set = pd.read_pickle(f'dataset/transfer_learning/{f}/train_sets/{sets[0]}').to_numpy()[:, :n_features]
            dev_set = pd.read_pickle(f'dataset/transfer_learning/{f}/dev_sets/{sets[1]}').to_numpy()[:, :n_features]

            train_ds = generate_inputs_outputs(train_set, past, horizon, batch_numbers, 1)
            dev_ds = generate_inputs_outputs(dev_set, past, horizon, batch_numbers, 1)

            i = 0
            while len(list(train_ds)) < 1:
                train_ds = generate_inputs_outputs(train_set, past, horizon, batch_numbers - i, 1)
                i += 1

            i = 0
            while len(list(dev_ds)) < 1:
                dev_ds = generate_inputs_outputs(dev_set, past, horizon, batch_numbers - i, 1)
                i += 1

            res = model.fit(x=train_ds, validation_data=dev_ds, epochs=epochs, shuffle=False,
                            callbacks=[early_stopping, reduce_lr])

            t1 = time.perf_counter()
            print(f'Time for {early_stopping.stopped_epoch} epochs:', t1 - t0)

        print(f'Done with {f}')

        os.makedirs(f'results/tests/transfer_learning/reduced_dim/{f}/', exist_ok=True)
        test_sets = os.listdir(f'dataset/transfer_learning/{f}/test_sets/')
        predictions_array = np.array([])
        true_array = np.array([])
        normalizer_y = load(f'dataset/transfer_learning/{f}/normalizer_y.joblib')
        for sets in test_sets:
            test_set = pd.read_pickle(f'dataset/transfer_learning/{f}/test_sets/{sets}').to_numpy()[:, :n_features]
            test_ds = generate_inputs_outputs(test_set, past, horizon, 128, 24)

            i = 0
            while len(list(test_ds)) < 1:
                test_ds = generate_inputs_outputs(test_set, past, horizon, 128 - i, 24)
                i += 1

            res = model.predict(test_ds)

            num = sets.replace('.', '_').split('_')[-2]
            _, output = list(test_ds)[0]
            for i in range(res.shape[0]):
                true_y = normalizer_y.inverse_transform(output[i, :].numpy().reshape(-1, 1))
                forecast_y = normalizer_y.inverse_transform(res[i, :].reshape(-1, 1))

                predictions_array = np.append(predictions_array, forecast_y.squeeze())
                true_array = np.append(true_array, true_y.squeeze())

        if true_array.shape[0] < 240:
            i = true_array.shape[0]
        else:
            i = 240

        fig, ax = plt.subplots(nrows=2, sharex=True)
        ax[0].plot(true_array[:i], label=r'$y$')
        ax[0].plot(predictions_array[:i], label=r'$\hat{y}$')
        ax[1].plot(np.abs(true_array - predictions_array)[:i])
        ax[0].legend()
        ax[0].set(ylabel=r'$PM_{2.5}$')
        ax[1].set(xlabel='Measurements', ylabel=r'$|y - \hat{y}|$')
        fig.savefig(f'results/tests/transfer_learning/reduced_dim/{f}/forecast_plots_individual_n{n_networks}.png')
        plt.close()

        mse = mean_squared_error(true_array, predictions_array)
        mae = mean_absolute_error(true_array, predictions_array)
        mpe = mean_absolute_percentage_error(true_array, predictions_array)
        metrics = {'Mean Squared Error': mse, 'Mean Absolute Error': mae, 'Mean Absolute Percentage Error': mpe}
        with open(f'results/tests/transfer_learning/reduced_dim/{f}/error_metrics_total_individual_n{n_networks}.csv', 'w') as error_file:
            w = csv.writer(error_file)
            for key, value in metrics.items():
                w.writerow([key, value])

    keras.backend.clear_session()

print('done with reduced_dims')
