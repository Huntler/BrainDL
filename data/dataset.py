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

        
        if self.length%self._seq > 0:
            print(f"Time steps {self.time_steps} is not divisible by {self._seq}!!!")

        self.length = int(self.length/self._seq)

        # check theory
        print(f"Iterated length  = {self.length}")
        th_length = int((len(self.files) * self.time_steps)/(self._seq))
        print(f"Theory length {th_length}")
        print(f"Number of all matrices {len(self.matrices)}")

    def preprocess_data(self):
        matrices = []
        labels =  []
        # Load all data into memory (if there are problems with RAM - just load them on request)
        for f in self.files:
            label = get_file_label(f)
            labels.append(label)

            matrix = get_dataset_matrix(f)

            if self.downsampling:
                matrix = self.downsample(matrix)
            
            if self.normalize:
                matrix = self.normalize_matrix(matrix)
            
            self.length = self.length + matrix.shape[1]
            matrices.append(matrix)

        return matrices, labels


    def scale_back(self, data):
        data = np.array(data, dtype=self._precision)
        return self._scaler.inverse_transform(data)

    def __onehot_ecnode(self, n: int) -> np.array:
        onehot = np.zeros((4, ), self._precision)
        onehot[n] = 1
        return onehot

    def __len__(self):
        return self.length
        #return len(self.files)

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

        # get index of matrix from which we will load data
        mat_index = int((index * self._seq)/self.time_steps)
        matrix = self.matrices[mat_index]

        # sum of time_steps until current matrix
        steps_to_matrix = mat_index * self.time_steps
        current_time_step = index * self._seq

        ts_index = current_time_step - steps_to_matrix

        # ts_index should be between 0 and self.time_steps
        if ts_index > self.time_steps:
            print(f"Matrix index {mat_index}")
            print(f"Steps to matrix {steps_to_matrix}")
            print(f"Time steps in a single matrix: {self.time_steps}")

            print(f"Index to get_item {index}")
            print(f"Current_time_step {current_time_step}")
            print(ts_index)
        
        # If ts_index + self._seq
        #if (ts_index + self._seq) > self.time_steps:
            

        # Get 2D meshes for self._seq number of time steps
        meshes = get_meshes(matrix, ts_index, self._seq)


        x = np.swapaxes(meshes, 0, 2)
        x = x.astype(self._precision)
        y = self.__onehot_ecnode(self.labels[mat_index])
        y = y[np.newaxis, :]
        y = np.repeat(y, self._seq, axis=0)
        return x, y


