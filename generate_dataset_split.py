import os
from joblib import dump
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def individual_dataset_split():
    files = os.listdir('dataset/merged/')
    for f in files:
        df_temp = pd.read_pickle(f'dataset/merged/{f}')

        num = df_temp.shape[0]
        split_1 = int(num*0.8)
        split_2 = int(num*0.9)
        train_set = df_temp.iloc[:split_1]
        dev_set = df_temp.iloc[split_1:split_2]
        test_set = df_temp.iloc[split_2:]

        cols = train_set.columns.drop(['LATITUDE', 'LONGITUDE', 'ELEVATION'])
        normalizer = MinMaxScaler()
        normalizer.fit(train_set[cols])

        train_set_temp = pd.DataFrame(normalizer.transform(train_set[cols]), index=train_set.index, columns=cols)
        train_set_temp = train_set_temp.assign(LATITUDE=train_set['LATITUDE'].values)
        train_set_temp = train_set_temp.assign(LONGITUDE=train_set['LONGITUDE'].values)
        train_set_temp = train_set_temp.assign(ELEVATION=train_set['ELEVATION'].values)
        train_set = train_set_temp.copy()
        del train_set_temp

        dev_set_temp = pd.DataFrame(normalizer.transform(dev_set[cols]), index=dev_set.index, columns=cols)
        dev_set_temp = dev_set_temp.assign(LATITUDE=dev_set['LATITUDE'].values)
        dev_set_temp = dev_set_temp.assign(LONGITUDE=dev_set['LONGITUDE'].values)
        dev_set_temp = dev_set_temp.assign(ELEVATION=dev_set['ELEVATION'].values)
        dev_set = dev_set_temp.copy()
        del dev_set_temp

        test_set_temp = pd.DataFrame(normalizer.transform(test_set[cols]), index=test_set.index, columns=cols)
        test_set_temp = test_set_temp.assign(LATITUDE=test_set['LATITUDE'].values)
        test_set_temp = test_set_temp.assign(LONGITUDE=test_set['LONGITUDE'].values)
        test_set_temp = test_set_temp.assign(ELEVATION=test_set['ELEVATION'].values)
        test_set = test_set_temp.copy()
        del test_set_temp

        train_set['group'] = train_set['pm25'].isna().cumsum()
        dev_set['group'] = dev_set['pm25'].isna().cumsum()
        test_set['group'] = test_set['pm25'].isna().cumsum()

        os.makedirs(f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/train_sets/', exist_ok=True)
        for idx, group in enumerate(train_set['group'].unique().tolist()):
            d = train_set[train_set['group'] == group]
            if d.shape[0] < 48:
                continue

            if (d.isna().sum() > 4).any():
                continue
            else:
                d = d.interpolate()

            d.to_pickle(f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/train_sets/train_set_{idx}.pkl')

        os.makedirs(f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/dev_sets/', exist_ok=True)
        for idx, group in enumerate(dev_set['group'].unique().tolist()):
            d = dev_set[dev_set['group'] == group]
            if d.shape[0] < 48:
                continue

            if (d.isna().sum() > 4).any():
                continue
            else:
                d = d.interpolate()

            d.to_pickle(f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/dev_sets/dev_set_{idx}.pkl')

        os.makedirs(f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/test_sets/', exist_ok=True)
        for idx, group in enumerate(test_set['group'].unique().tolist()):
            d = test_set[test_set['group'] == group]
            if d.shape[0] < 48:
                continue

            if (d.isna().sum() > 4).any():
                continue
            else:
                d = d.interpolate()

            d.to_pickle(f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/test_sets/test_set_{idx}.pkl')

        dump(normalizer, f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/normalizer.joblib')

        print(f'done with {f.split("_")[0]}')

    print('done')


def grouped_dataset_split(rng_int: int):
    rng = np.random.default_rng(rng_int)
    files = os.listdir('dataset/merged/')
    set_x = []
    set_y = []
    for f in files:
        df_temp = pd.read_pickle(f'dataset/merged/{f}')
        df_temp = df_temp.interpolate()

        min_index = df_temp.index[0]
        if min_index.hour != 0:
            min_index = df_temp.index[df_temp.index.hour == 0][0]

        max_index = df_temp.index[-1]
        if max_index.hour != 0:
            max_index = df_temp.index[df_temp.index.hour == 0][-1]

        cal = np.arange(min_index, max_index, step=np.timedelta64(1, 'D'), dtype='datetime64')
        n = np.floor(cal.shape[0]/2*0.9).astype(int)
        cal = rng.choice(cal, n, replace=False, shuffle=False)
        for day in cal:
            if df_temp.loc[day:(day + np.timedelta64(2, 'D')), :].shape[0] < 48:
                continue

            if (df_temp.loc[day:(day + np.timedelta64(2, 'D')), :].isna().sum() > 4).any():
                continue
            else:
                x_df = df_temp.loc[day:(day + np.timedelta64(2, 'D')), :].interpolate()

            if (df_temp.loc[(day + np.timedelta64(1, 'D')):(day + np.timedelta64(47, 'h')), 'pm25'].isna().sum() > 4).any():
                continue
            else:
                y_df = df_temp.loc[(day + np.timedelta64(1, 'D')):(day + np.timedelta64(47, 'h')), 'pm25'].interpolate()

            set_x.append(x_df)
            set_y.append(y_df.values)

        print(f'Approx. number of {f.split("_")[0]} elements: {n}')

    set_x = np.array(set_x)
    set_y = np.array(set_y)
    set_y = set_y.reshape(set_y.shape[0], set_y.shape[1], 1)

    num = set_x.shape[0]
    split_1 = int(num*0.8)
    split_2 = int(num*0.9)
    shuffle = rng.permutation(num)
    set_x = set_x[shuffle, :, :]
    set_y = set_y[shuffle, :, :]

    train_set_x, train_set_y = set_x[:split_1], set_y[:split_1]
    dev_set_x, dev_set_y = set_x[split_1:split_2], set_y[split_1:split_2]
    test_set_x, test_set_y = set_x[split_2:], set_y[split_2:]

    normalizer_x = MinMaxScaler()
    normalizer_y = MinMaxScaler()
    train_set_x[:, :, 0] = normalizer_x.fit_transform(train_set_x[:, :, 0])
    train_set_y[:, :, 0] = normalizer_y.fit_transform(train_set_y.squeeze())
    dev_set_x[:, :, 0] = normalizer_x.transform(dev_set_x[:, :, 0])
    dev_set_y[:, :, 0] = normalizer_y.transform(dev_set_y[:, :, 0])
    test_set_x[:, :, 0] = normalizer_x.transform(test_set_x[:, :, 0])
    test_set_y[:, :, 0] = normalizer_y.transform(test_set_y[:, :, 0])

    dump(normalizer_x, 'dataset/lstm_dataset_splits/collective/normalizer_x.joblib')
    dump(normalizer_y, 'dataset/lstm_dataset_splits/collective/normalizer_y.joblib')

    for i in range(1, train_set_x.shape[2]):
        train_set_x[:, :, i] = normalizer_x.transform(train_set_x[:, :, i])
        dev_set_x[:, :, i] = normalizer_x.transform(dev_set_x[:, :, i])
        test_set_x[:, :, i] = normalizer_x.transform(test_set_x[:, :, i])

    np.save('dataset/lstm_dataset_splits/collective/train_set_x', train_set_x)
    np.save('dataset/lstm_dataset_splits/collective/train_set_y', train_set_y)
    np.save('dataset/lstm_dataset_splits/collective/dev_set_x', dev_set_x)
    np.save('dataset/lstm_dataset_splits/collective/dev_set_y', dev_set_y)
    np.save('dataset/lstm_dataset_splits/collective/test_set_x', test_set_x)
    np.save('dataset/lstm_dataset_splits/collective/test_set_y', test_set_y)

    print('done')


def transfer_dataset_split():
    files = os.listdir('dataset/merged/')
    for f in files:
        df_temp = pd.read_pickle(f'dataset/merged/{f}')

        num = df_temp.shape[0]
        split_1 = int(num*0.8)
        split_2 = int(num*0.9)
        train_set = df_temp.iloc[:split_1]
        dev_set = df_temp.iloc[split_1:split_2]
        test_set = df_temp.iloc[split_2:]

        cols = train_set.columns.drop(['LATITUDE', 'LONGITUDE', 'ELEVATION'])
        normalizer = MinMaxScaler()
        normalizer.fit(train_set[cols])

        train_set_temp = pd.DataFrame(normalizer.transform(train_set[cols]), index=train_set.index, columns=cols)
        train_set_temp = train_set_temp.assign(LATITUDE=train_set['LATITUDE'].values)
        train_set_temp = train_set_temp.assign(LONGITUDE=train_set['LONGITUDE'].values)
        train_set_temp = train_set_temp.assign(ELEVATION=train_set['ELEVATION'].values)
        train_set = train_set_temp.copy()
        del train_set_temp

        dev_set_temp = pd.DataFrame(normalizer.transform(dev_set[cols]), index=dev_set.index, columns=cols)
        dev_set_temp = dev_set_temp.assign(LATITUDE=dev_set['LATITUDE'].values)
        dev_set_temp = dev_set_temp.assign(LONGITUDE=dev_set['LONGITUDE'].values)
        dev_set_temp = dev_set_temp.assign(ELEVATION=dev_set['ELEVATION'].values)
        dev_set = dev_set_temp.copy()
        del dev_set_temp

        test_set_temp = pd.DataFrame(normalizer.transform(test_set[cols]), index=test_set.index, columns=cols)
        test_set_temp = test_set_temp.assign(LATITUDE=test_set['LATITUDE'].values)
        test_set_temp = test_set_temp.assign(LONGITUDE=test_set['LONGITUDE'].values)
        test_set_temp = test_set_temp.assign(ELEVATION=test_set['ELEVATION'].values)
        test_set = test_set_temp.copy()
        del test_set_temp

        train_set['group'] = train_set['pm25'].isna().cumsum()
        dev_set['group'] = dev_set['pm25'].isna().cumsum()
        test_set['group'] = test_set['pm25'].isna().cumsum()

        os.makedirs(f'dataset/transfer_learning/{f.split("_")[0]}/train_sets/', exist_ok=True)
        for idx, group in enumerate(train_set['group'].unique().tolist()):
            d = train_set[train_set['group'] == group]
            if d.shape[0] < 48:
                continue

            if (d.isna().sum() > 4).any():
                continue
            else:
                d = d.interpolate()

            d.to_pickle(f'dataset/transfer_learning/{f.split("_")[0]}/train_sets/train_set_{idx}.pkl')

        os.makedirs(f'dataset/transfer_learning/{f.split("_")[0]}/dev_sets/', exist_ok=True)
        for idx, group in enumerate(dev_set['group'].unique().tolist()):
            d = dev_set[dev_set['group'] == group]
            if d.shape[0] < 48:
                continue

            if (d.isna().sum() > 4).any():
                continue
            else:
                d = d.interpolate()

            d.to_pickle(f'dataset/transfer_learning/{f.split("_")[0]}/dev_sets/dev_set_{idx}.pkl')

        os.makedirs(f'dataset/transfer_learning/{f.split("_")[0]}/test_sets/', exist_ok=True)
        for idx, group in enumerate(test_set['group'].unique().tolist()):
            d = test_set[test_set['group'] == group]
            if d.shape[0] < 48:
                continue

            if (d.isna().sum() > 4).any():
                continue
            else:
                d = d.interpolate()

            d.to_pickle(f'dataset/transfer_learning/{f.split("_")[0]}/test_sets/test_set_{idx}.pkl')

        dump(normalizer, f'dataset/transfer_learning/{f.split("_")[0]}/normalize.joblib')

        print(f'done with {f.split("_")[0]}')

    print('done')


individual_dataset_split()
grouped_dataset_split(0)
transfer_dataset_split()
