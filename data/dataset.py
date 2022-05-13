from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import torch
import numpy as np
import os
import h5py


# the dataset has t time steps recorded with 248 features at each sample
# I would normalize each feature on its own and not all 248 together
# otherwise, some values might get too small

class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, d_type: str = "train", normalize: bool = True, bounds: Tuple[int] = (0, 1),
                future_steps: int= 1, sequence_length: int = 1, precision: np.dtype = np.float32,
                task_dir: str = "./data/data/Intra", task_type: str="Intra"):
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
            for f in os.listdir(d):
                p = os.path.join(d,f)
                self.files.append(os.path.abspath(p))

        self._mat = None

        self.normalize = normalize
        # normalize the dataset between values of o to 1
        self._scaler = None
        if normalize:
            self._scaler = MinMaxScaler(feature_range=bounds)

        '''
        # load the dataset specified
        self._mat = scipy.io.loadmat(self._file).get(f"X{d_type}")
        self._mat = self._mat.astype(self._precision)

            self._scaler.fit(self._mat)
            self._mat = self. _scaler.transform(self._mat)

        '''
    def __get_dataset_name(self,file_name_with_dir: str) -> str:
        filename_without_dir = file_name_with_dir.split('/')[-1]
        temp = filename_without_dir.split('_')[:-1]
        dataset_name = "_".join(temp)
        return dataset_name

    def get_dataset_matrix(self,file_path: str) -> np.array:
        with h5py.File(file_path,'r') as f:
            dataset_name = self.__get_dataset_name(file_path)
            matrix = f.get(dataset_name)[()]
            return matrix


    def scale_back(self, data):
        data = np.array(data, dtype=self._precision)
        return self._scaler.inverse_transform(data)

    def __len__(self):
        return len(self.files)

    
    def get_file_label(self,file):
        name = os.path.basename(file)
        if "rest" in name:
            return 0
        elif "task_motor" in name:
            return 1
        elif "task_story" in name:
            return 2
        else:
            return 3
            

    def __getitem__(self, index):
        self._mat = self.get_dataset_matrix(self.files[index])

        if self.normalize:
            self._scaler.fit(self._mat)
            self._mat = self. _scaler.transform(self._mat)

        X = self._mat

        label = self.get_file_label(self.files[index])

        y = np.full(X.shape[0], label)
        return X, y
