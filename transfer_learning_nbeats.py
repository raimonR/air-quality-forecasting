import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mape, mse
import torch


model = NBEATSmodel.load_model('dataset/transfer_learning/transfer_learning_model.pth.tar')

for param in model.parameters():
    print(param)
