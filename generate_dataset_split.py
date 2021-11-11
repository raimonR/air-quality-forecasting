import os
from joblib import dump
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# TODO: NEW NORMALIZATION METHOD IS NOT GOOD (FOR ALL DATASET SPLITS), SHOULD GO BACK TO OLD ONE --> CHECK GITHUB COMMITS.
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

        normalizer = StandardScaler()
        normalizer_y = StandardScaler()
        train_set = train_set.assign(pm25=normalizer_y.fit_transform(train_set['pm25'].to_numpy().reshape(-1, 1)))
        dev_set = dev_set.assign(pm25=normalizer_y.transform(dev_set['pm25'].to_numpy().reshape(-1, 1)))
        test_set = test_set.assign(pm25=normalizer_y.transform(test_set['pm25'].to_numpy().reshape(-1, 1)))
        for col in train_set.columns.to_list()[4:18]:
            kwargs = {col: normalizer.fit_transform(train_set[col].to_numpy().reshape(-1, 1))}
            train_set = train_set.assign(**kwargs)
            kwargs = {col: normalizer.transform(dev_set[col].to_numpy().reshape(-1, 1))}
            dev_set = dev_set.assign(**kwargs)
            kwargs = {col: normalizer.transform(test_set[col].to_numpy().reshape(-1, 1))}
            test_set = test_set.assign(**kwargs)

        train_scale_temp = normalizer.fit_transform(train_set.iloc[:, -450:-300])
        dev_scale_temp = normalizer.transform(dev_set.iloc[:, -450:-300])
        test_scale_temp = normalizer.transform(test_set.iloc[:, -450:-300])
        for i, col in enumerate(train_set.columns.to_list()[-450:-300]):
            kwargs = {col: train_scale_temp}
            train_set = train_set.assign(**kwargs)
            kwargs = {col: dev_scale_temp}
            dev_set = dev_set.assign(**kwargs)
            kwargs = {col: test_scale_temp}
            test_set = test_set.assign(**kwargs)

        train_scale_temp = normalizer.fit_transform(train_set.iloc[:, -300:-150])
        dev_scale_temp = normalizer.transform(dev_set.iloc[:, -300:-150])
        test_scale_temp = normalizer.transform(test_set.iloc[:, -300:-150])
        for i, col in enumerate(train_set.columns.to_list()[-300:-150]):
            kwargs = {col: train_scale_temp}
            train_set = train_set.assign(**kwargs)
            kwargs = {col: dev_scale_temp}
            dev_set = dev_set.assign(**kwargs)
            kwargs = {col: test_scale_temp}
            test_set = test_set.assign(**kwargs)

        train_scale_temp = normalizer.fit_transform(train_set.iloc[:, -150:])
        dev_scale_temp = normalizer.transform(dev_set.iloc[:, -150:])
        test_scale_temp = normalizer.transform(test_set.iloc[:, -150:])
        for i, col in enumerate(train_set.columns.to_list()[-150:]):
            kwargs = {col: train_scale_temp}
            train_set = train_set.assign(**kwargs)
            kwargs = {col: dev_scale_temp}
            dev_set = dev_set.assign(**kwargs)
            kwargs = {col: test_scale_temp}
            test_set = test_set.assign(**kwargs)

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

        dump(normalizer_y, f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/normalizer_y.joblib')

        print(f'done with {f.split("_")[0]}')

    print('done')


def grouped_dataset_split(rng_int: int):
    rng = np.random.default_rng(rng_int)
    files = os.listdir('dataset/merged/')
    set_x = []
    set_y = []
    for f in files:
        df_temp = pd.read_pickle(f'dataset/merged/{f}')

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

    normalizer_x = StandardScaler()
    normalizer_y = StandardScaler()
    train_set_y[:, :, 0] = normalizer_y.fit_transform(train_set_y.squeeze())
    dev_set_y[:, :, 0] = normalizer_y.transform(dev_set_y[:, :, 0])
    test_set_y[:, :, 0] = normalizer_y.transform(test_set_y[:, :, 0])

    for i in range(18):
        train_set_x[:, :, i] = normalizer_x.fit_transform(train_set_x[:, :, i])
        dev_set_x[:, :, i] = normalizer_x.transform(dev_set_x[:, :, i])
        test_set_x[:, :, i] = normalizer_x.transform(test_set_x[:, :, i])

    for i in range(train_set_x.shape[1]):
        train_set_x[:, i, -450:-300] = normalizer_x.fit_transform(train_set_x[:, i, -450:-300])
        dev_set_x[:, i, -450:-300] = normalizer_x.transform(dev_set_x[:, i, -450:-300])
        test_set_x[:, i, -450:-300] = normalizer_x.transform(test_set_x[:, i, -450:-300])

        train_set_x[:, i, -300:-150] = normalizer_x.fit_transform(train_set_x[:, i, -300:-150])
        dev_set_x[:, i, -300:-150] = normalizer_x.transform(dev_set_x[:, i, -300:-150])
        test_set_x[:, i, -300:-150] = normalizer_x.transform(test_set_x[:, i, -300:-150])

        train_set_x[:, i, -150:] = normalizer_x.fit_transform(train_set_x[:, i, -150:])
        dev_set_x[:, i, -150:] = normalizer_x.transform(dev_set_x[:, i, -150:])
        test_set_x[:, i, -150:] = normalizer_x.transform(test_set_x[:, i, -150:])

    dump(normalizer_y, 'dataset/lstm_dataset_splits/collective/normalizer_y.joblib')

    train_set_x = np.nan_to_num(train_set_x)
    train_set_y = np.nan_to_num(train_set_y)
    dev_set_x = np.nan_to_num(dev_set_x)
    dev_set_y = np.nan_to_num(dev_set_y)
    test_set_x = np.nan_to_num(test_set_x)
    test_set_y = np.nan_to_num(test_set_y)

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

        normalizer = StandardScaler()
        normalizer_y = StandardScaler()
        train_set = train_set.assign(pm25=normalizer_y.fit_transform(train_set['pm25'].to_numpy().reshape(-1, 1)))
        dev_set = dev_set.assign(pm25=normalizer_y.transform(dev_set['pm25'].to_numpy().reshape(-1, 1)))
        test_set = test_set.assign(pm25=normalizer_y.transform(test_set['pm25'].to_numpy().reshape(-1, 1)))
        for col in train_set.columns.to_list()[4:18]:
            kwargs = {col: normalizer.fit_transform(train_set[col].to_numpy().reshape(-1, 1))}
            train_set = train_set.assign(**kwargs)
            kwargs = {col: normalizer.transform(dev_set[col].to_numpy().reshape(-1, 1))}
            dev_set = dev_set.assign(**kwargs)
            kwargs = {col: normalizer.transform(test_set[col].to_numpy().reshape(-1, 1))}
            test_set = test_set.assign(**kwargs)

        train_scale_temp = normalizer.fit_transform(train_set.iloc[:, -450:-300])
        dev_scale_temp = normalizer.transform(dev_set.iloc[:, -450:-300])
        test_scale_temp = normalizer.transform(test_set.iloc[:, -450:-300])
        for i, col in enumerate(train_set.columns.to_list()[-450:-300]):
            kwargs = {col: train_scale_temp}
            train_set = train_set.assign(**kwargs)
            kwargs = {col: dev_scale_temp}
            dev_set = dev_set.assign(**kwargs)
            kwargs = {col: test_scale_temp}
            test_set = test_set.assign(**kwargs)

        train_scale_temp = normalizer.fit_transform(train_set.iloc[:, -300:-150])
        dev_scale_temp = normalizer.transform(dev_set.iloc[:, -300:-150])
        test_scale_temp = normalizer.transform(test_set.iloc[:, -300:-150])
        for i, col in enumerate(train_set.columns.to_list()[-300:-150]):
            kwargs = {col: train_scale_temp}
            train_set = train_set.assign(**kwargs)
            kwargs = {col: dev_scale_temp}
            dev_set = dev_set.assign(**kwargs)
            kwargs = {col: test_scale_temp}
            test_set = test_set.assign(**kwargs)

        train_scale_temp = normalizer.fit_transform(train_set.iloc[:, -150:])
        dev_scale_temp = normalizer.transform(dev_set.iloc[:, -150:])
        test_scale_temp = normalizer.transform(test_set.iloc[:, -150:])
        for i, col in enumerate(train_set.columns.to_list()[-150:]):
            kwargs = {col: train_scale_temp}
            train_set = train_set.assign(**kwargs)
            kwargs = {col: dev_scale_temp}
            dev_set = dev_set.assign(**kwargs)
            kwargs = {col: test_scale_temp}
            test_set = test_set.assign(**kwargs)

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

        dump(normalizer_y, f'dataset/lstm_dataset_splits/individual/{f.split("_")[0]}/normalizer_y.joblib')

        print(f'done with {f.split("_")[0]}')

    print('done')


individual_dataset_split()
grouped_dataset_split(0)
transfer_dataset_split()
