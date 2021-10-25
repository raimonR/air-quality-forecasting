import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense

files = os.listdir('dataset/lstm_dataset_splits/individual/')
for f in files:
    train_set_x = np.load(f'dataset/lstm_dataset_splits/individual/{f}/train_set_x.npy')
    train_set_y = np.load(f'dataset/lstm_dataset_splits/individual/{f}/train_set_y.npy')
    dev_set_x = np.load(f'dataset/lstm_dataset_splits/individual/{f}/dev_set_x.npy')
    dev_set_y = np.load(f'dataset/lstm_dataset_splits/individual/{f}/dev_set_y.npy')
    test_set_x = np.load(f'dataset/lstm_dataset_splits/individual/{f}/test_set_x.npy')
    test_set_y = np.load(f'dataset/lstm_dataset_splits/individual/{f}/test_set_y.npy')


