from pkgutil import get_data
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scipy.io
import torch
import numpy as np
import os
import h5py

from .utils import get_dataset_matrix, get_file_label, get_meshes

# the dataset has t time steps recorded with 248 features at each sample
# I would normalize each feature on its own and not all 248 together
# otherwise, some values might get too small

class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, d_type: str = "train", normalize: bool = True, bounds: Tuple[int] = (0, 1),
                future_steps: int= 1, sequence_length: int = 1, precision: np.dtype = np.float32,
                task_dir: str = "./data/data/Intra", task_type: str="Intra", 
                global_normalization: bool = True, 
                zscore_normalization: bool = False,
                downsampling: bool = False, downsample_by: float = 0.5):
        super(BrainDataset, self).__init__()

        self._precision = precision
        self._seq = sequence_length
        self._f_seq = future_steps
        self._d_type = d_type

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

        self.normalize = normalize
        self.global_normalization = global_normalization
        self.zscore_normalization = zscore_normalization

        self.downsampling = downsampling
        self.downsample_by = downsample_by

        # normalize the dataset between values of o to 1
        self._scaler = None
        if self.normalize:
            if self.zscore_normalization:
                self._scaler = StandardScaler()
            else:
                self._scaler = MinMaxScaler(feature_range=bounds)
        
        # Number of timesteps in each file that will be taken
        self.time_steps = 35624
        if self.downsampling:
            self.time_steps = int(self.time_steps * (1.0 - self.downsample_by))

        # in theory: Length = (number of files * time_steps * downsample_by)/sequence length 
        self.length = 0

        # We will load all data and do the downsampling + normalization at initialization
        self.matrices, self.labels = self.preprocess_data()

        

    def preprocess_data(self):
        matrices = []
        labels =  []
        # Load all data into memory (if there are problems with RAM - just load them on request)
        i = 0
        for f in self.files:
            label = get_file_label(f)

            matrix = get_dataset_matrix(f)

            if self.downsampling:
                matrix = self.downsample(matrix)
            
            if self.normalize:
                matrix = self.normalize_matrix(matrix)
            
            self.length = self.length + matrix.shape[1]


            if i == 0:
                matrices.append(matrix)
                labels.append(label)
                i = 1
                continue
            
            # find index of label in labels
            ind = -1
            if label in labels:
                ind = labels.index(label)

            if ind == -1:
                matrices.append(matrix)
                labels.append(label)
                continue
            
            # stack the matrix to the appropriate label
            matrices[ind] = np.hstack((matrices[ind], matrix))

        return matrices, labels


    def scale_back(self, data):
        data = np.array(data, dtype=self._precision)
        return self._scaler.inverse_transform(data)

    def __onehot_ecnode(self, n: int) -> np.array:
        onehot = np.zeros((4, ), self._precision)
        onehot[n] = 1
        return onehot

    def __len__(self):
        return int(self.length)

    def normalize_globally(self,matrix):
        # Normalize all cells together throughout all time steps
        # with either MinMax or Standard scaler
        # Shape of matrix is (248, 35624) -> 248 rows for each sensor and 35624 time steps

        #print(f'Normalizing globally matrix of shape {matrix.shape}')

        # scaler scales each feature/column so we have to convert matrix to 1D
        # to scale based on all sensors in all time steps together
        shape = matrix.shape
        matrix = matrix.reshape((shape[0]*shape[1], 1))
        
        self._scaler.fit(matrix)
        self._scaler.transform(matrix)

        matrix = matrix.reshape(shape)
        return matrix

    def normalize_locally(self,matrix):
        # Normalize each cell individually throughout all time steps
        # with either MinMax or Standard scaler
        # Shape of matrix is (248, 35624) -> 248 rows for each sensor and 35624 time steps

        #print(f'Normalizing locally matrix of shape {matrix.shape}')

        matrix = matrix.transpose() # scaler scales each feature/column so we have to transpose to 
                                    # scale each sensor individually
        self._scaler.fit(matrix)
        self._scaler.transform(matrix)

        matrix = matrix.transpose()
        return matrix

    def normalize_matrix(self, matrix):
        if self.global_normalization:
            matrix = self.normalize_globally(matrix)
        else:
            matrix = self.normalize_locally(matrix)
        return matrix

    def downsample(self, matrix):
        # Downsample the matrix by just deleting columns on a uniform distribution
        shape = matrix.shape
        matrix_time_steps = shape[1]

        if self.downsample_by > 1.0:
            print(f'downsample_by must be between 0 and 1!')
            return

        # We only want self.time_steps number of time steps after deletion
        # num _of_samples represents the amount of samples we will delete
        num_of_samples = matrix_time_steps - self.time_steps
        
        to_delete = np.linspace(0,matrix_time_steps-1,num_of_samples, dtype=int)
        matrix  = np.delete(matrix, to_delete, axis=1)

        return matrix

    def __getitem__(self, index):
        # We return a sequence of meshes from [index,index+self._seq] from 
        # the appropriate matrix in self.matrices

        length = self.matrices[0].shape[1]
        selected_matrix = self.matrices[index // length]
        label = self.labels[0]

        # calculate the correct relative start index
        rel_start = index % (length - self._seq)

        # Get 2D meshes for self._seq number of time steps
        meshes = get_meshes(selected_matrix, rel_start, self._seq)

        if self._d_type == "train":
            meshes = meshes + np.random.normal(0, 1, meshes.shape)

        x = meshes.astype(self._precision)
        y = np.array([label], dtype=np.uint8)
        return x, y


