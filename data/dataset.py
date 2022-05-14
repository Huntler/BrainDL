from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import torch
import numpy as np
import os
import h5py

from .utils import get_dataset_matrix, get_meshes

# the dataset has t time steps recorded with 248 features at each sample
# I would normalize each feature on its own and not all 248 together
# otherwise, some values might get too small

class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, d_type: str = "train", normalize: bool = True, bounds: Tuple[int] = (0, 1),
                future_steps: int= 1, sequence_length: int = 1, precision: np.dtype = np.float32,
                task_dir: str = "./data/data/Intra", task_type: str="Intra", global_normalization: bool = True, 
                downsampling: int = 1.0):
        super(BrainDataset, self).__init__()

        self._precision = precision
        self._seq = sequence_length
        self._f_seq = future_steps

        # Get all directories of d_type in task_dir
        dirs = []
        for file in os.listdir(task_dir):
            d = os.path.join(task_dir, file)
            if os.path.isdir(d):
                dirs.append(d)
        
        # Get all file paths
        self.files = []
        for d in dirs:
            if d_type not in d:
                continue
            for f in os.listdir(d):
                p = os.path.join(d,f)
                self.files.append(os.path.abspath(p))

        self._mat = None


        self.normalize = normalize
        self.global_normalization = global_normalization
        self.downsampling = downsampling

        # normalize the dataset between values of o to 1
        self._scaler = None
        if self.normalize:
            self._scaler = MinMaxScaler(feature_range=bounds)

        '''
        # load the dataset specified
        self._mat = scipy.io.loadmat(self._file).get(f"X{d_type}")
        self._mat = self._mat.astype(self._precision)

            self._scaler.fit(self._mat)
            self._mat = self. _scaler.transform(self._mat)

        '''


    def scale_back(self, data):
        data = np.array(data, dtype=self._precision)
        return self._scaler.inverse_transform(data)

    def __len__(self):
        return len(self.files)


            

    def __getitem__(self, index):
        file = self.files[index]
        matrix = get_dataset_matrix(file)
        

        # TODO: normalization + downsampling + sequencing


        time_steps = 10
        # Get 2D meshes for 10 time steps -> sequencing stuff
        meshes = get_meshes(matrix, time_steps)

        # normalization 
        '''
        if self.normalize:
            self._scaler.fit(self._mat)
            self._mat = self. _scaler.transform(self._mat)        
        '''

        #X,y = preprocess_data_type(self._mat, 10,1) 
        #y = y*get_file_label(file)

        return meshes


