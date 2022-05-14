from typing import List
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
import numpy as np

from .utils import get_dataset_matrix


def reshape_matrix(matrix: np.array) -> np.array:
    """Method reshapes the given matrix with a shape of (248, *) to a shape of
    (8, 31).

    Args:
        matrix (np.array): The matrix with an input shape of (248, *)

    Returns:
        np.array: The matrix with a shape of (8, 31).
    """
    matrix = np.mean(matrix, axis=1)
    matrix = np.reshape(matrix, (8, 31))
    return matrix

def plot_matrix(matrix: np.array, name: str = None, disk_path: str = None) -> None | AxesImage:
    """This method plots a given matrix using imshow.

    Args:
        matrix (np.array): The matrix to plot.
        disk_path (str, optional): The plot is stored to the provided path if given. Defaults to None.

    Returns:
        None | AxesImage: None if the plot was saved to disk, otherwise the AxesImage object.
    """
    # reshape the matrix a bit to show its structure
    matrix = reshape_matrix(matrix)

    # create the image
    image = plt.imshow(matrix)
    if name:
        plt.title(name)

    if disk_path:
        plt.savefig(disk_path, image)
        return None

    return image

def plot_matrices(matrices: List[np.array], names: List[str] = [], disk_path: str = None) -> None | AxesImage:
    """This method plots multiple matrices.

    Args:
        matrices (List[np.array]): The list of matrices to show.
        names (List[str], optional): The name of each matrix. Defaults to [].
        disk_path (str, optional): The path to store the plot. Defaults to None.

    Returns:
        None | AxesImage: None if the plot was saved to disk, otherwise the AxesImage object.
    """
    fig = plt.figure(figsize=(2, len(matrices)))
    for i in range(1, len(matrices) + 1):
        matrix = reshape_matrix(matrices[i - 1])
        ax = fig.add_subplot(len(matrices), 1, i)
        if len(names) == len(matrices):
            ax.set_title(names[i - 1])
        plt.imshow(matrix)

    plt.tight_layout()

    # create the image
    if disk_path:
        plt.savefig(disk_path, fig)
        return None

    return fig


if __name__ == "__main__":
    # test this file
    path = "data/Data_Ass3/Intra/train/rest_105923_1.h5"
    matrix_0 = get_dataset_matrix(path)

    path = "data/Data_Ass3/Intra/train/task_story_math_105923_1.h5"
    matrix_1 = get_dataset_matrix(path)

    path = "data/Data_Ass3/Intra/train/task_working_memory_105923_1.h5"
    matrix_2 = get_dataset_matrix(path)

    path = "data/Data_Ass3/Intra/train/task_motor_105923_1.h5"
    matrix_3 = get_dataset_matrix(path)
    
    plot_matrices(matrices=[matrix_0, matrix_1, matrix_2, matrix_3], names=["Rest", "Math", "Memory", "Motor"])
    plt.show()

    # showing different chunks of the same class
    path = "data/Data_Ass3/Intra/train/task_motor_105923_1.h5"
    matrix_0 = get_dataset_matrix(path)

    path = "data/Data_Ass3/Intra/train/task_motor_105923_2.h5"
    matrix_1 = get_dataset_matrix(path)

    path = "data/Data_Ass3/Intra/train/task_motor_105923_3.h5"
    matrix_2 = get_dataset_matrix(path)

    path = "data/Data_Ass3/Intra/train/task_motor_105923_4.h5"
    matrix_3 = get_dataset_matrix(path)
    
    plot_matrices(matrices=[matrix_0, matrix_1, matrix_2, matrix_3], names=["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4"])
    plt.show()