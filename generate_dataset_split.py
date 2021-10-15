import pandas as pd
import numpy as np
import os
import sklearn as skl


def ensemble_dataset_split(rng_int: int, radiosonde_graphs: bool):
    rng = np.random.default_rng(rng_int)

    if radiosonde_graphs:
        files = os.listdir('dataset/merged/graphs/')
        files = ['graphs/' + f for f in files]
    else:
        files = os.listdir('dataset/merged/variables/')
        files = ['variables/' + f for f in files]

    for f in files:
        if f == 'Abidjan_merged_dataset.pkl':
            df_temp = pd.read_pickle(f'dataset/merged/{f}')
            min_index = df_temp.index[0]
            max_index = df_temp.index[-1]
            abidjan_cal = np.arange(min_index, max_index, step=np.timedelta64(1, 'D'), dtype='datetime64')

            intersection = df_temp.isna()

        elif f == 'Chennai_merged_dataset.pkl':
            df_temp = pd.read_pickle(f'dataset/merged/{f}')
            min_index = df_temp.index[0]
            max_index = df_temp.index[-1]
            chennai_cal = np.arange(min_index, max_index, step=np.timedelta64(1, 'D'), dtype='datetime64')
        elif f == 'Melbourne_merged_dataset.pkl':
            df_temp = pd.read_pickle(f'dataset/merged/{f}')
            min_index = df_temp.index[0]
            max_index = df_temp.index[-1]
            melbourne_cal = np.arange(min_index, max_index, step=np.timedelta64(1, 'D'), dtype='datetime64')
        elif f == 'Munich_merged_dataset.pkl':
            df_temp = pd.read_pickle(f'dataset/merged/{f}')
            min_index = df_temp.index[0]
            max_index = df_temp.index[-1]
            munich_cal = np.arange(min_index, max_index, step=np.timedelta64(1, 'D'), dtype='datetime64')
        elif f == 'Oakland_merged_dataset.pkl':
            df_temp = pd.read_pickle(f'dataset/merged/{f}')
            min_index = df_temp.index[0]
            max_index = df_temp.index[-1]
            oakland_cal = np.arange(min_index, max_index, step=np.timedelta64(1, 'D'), dtype='datetime64')
        elif f == 'Santiago_merged_dataset.pkl':
            df_temp = pd.read_pickle(f'dataset/merged/{f}')
            min_index = df_temp.index[0]
            max_index = df_temp.index[-1]
            santiago_cal = np.arange(min_index, max_index, step=np.timedelta64(1, 'D'), dtype='datetime64')



def transfer_dataset_split(rng_int: int):
    rng = np.random.default_rng(rng_int)


ensemble_dataset_split(0, True)
