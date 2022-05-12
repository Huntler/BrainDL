import numpy as np
import h5py

def __get_dataset_name(file_name_with_dir: str) -> str:
    filename_without_dir = file_name_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

def get_dataset_matrix(file_path: str) -> np.array:
    with h5py.File(file_path,'r') as f:
        dataset_name = __get_dataset_name(file_path)
        matrix = f.get(dataset_name)[()]
        return matrix