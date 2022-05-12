import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# the dataset has t time steps recorded with 248 features at each sample
# I would normalize each feature on its own and not all 248 together
# otherwise, some values might get too small


class BrainDataset(torch.utils.data.Dataset):
    # a (simple) usage of the parent class can be found here: ./submodules/data/dataset.py
    pass