import argparse
from multiprocessing import freeze_support
import torch
from model.brain_behaviour_classifier import BrainBehaviourClassifier
from submodules.TimeSeriesDL.utils.config import config
import math
import copy
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from data.dataset import BrainDataset

# some general brain information and proposed models: https://arxiv.org/pdf/2201.04229.pdf

config.register_model("BrainBehaviourClassifier", BrainBehaviourClassifier)


config_dict = None

def train():
    # define parameters (depending on device, lower the precision to save memory)
    device = config_dict["device"]
    precision = torch.float16 if device == "cuda" else torch.float32

    freeze_support()

    future_steps = config_dict["dataset_args"]["future_steps"]

    # load the data, normalize them and convert them to tensor
    dataset = BrainDataset(**config_dict["dataset_args"])
    
    split_sizes = [int(math.ceil(len(dataset) * 0.8)), int(math.floor(len(dataset) * 0.2))]
    
    trainset, valset = torch.utils.data.random_split(dataset, split_sizes)
    trainloader = DataLoader(trainset, **config_dict["dataloader_args"])
    valloader = DataLoader(valset, **config_dict["dataloader_args"])
    

def testing():
    # create a dummy input
    # batch_size, sequence length, sample (248 values reshaped to 8x31)
    x = torch.zeros((64, 4, 8, 31))

    model = BrainBehaviourClassifier()
    out = model(x)


def plot_matrix(matrix):
    H = np.array(matrix)
    plt.imshow(H, interpolation='none')
    plt.show()


def visualize_data():
    # load the data, normalize them and convert them to tensor
    dataset = BrainDataset(**config_dict["dataset_args"])
    print(f'Dataset length: {len(dataset)}')

    dataloader = DataLoader(dataset, **config_dict["dataloader_args"])

    dataiter = iter(dataloader)
    data = dataiter.next()


    mat = data[0,:,:,0]
    plot_matrix(mat)



if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="This program trains and tests a deep " + 
                                     "learning model to detect a behaviour based on brain data")
    parser.add_argument("--config", dest="config", help="Set path to config file.")
    parser.add_argument("--debug", dest="debug", help="Runs the model with a dummy input to debug it.", action="store_true")
    parser.add_argument("--visualize", dest="visual", help="Loads the data and visualizes it.", action="store_true")

    args = parser.parse_args()

        
    if args.debug:
        testing()
        quit()

    if args.visual:
        config_dict = config.get_args(args.config)
        visualize_data()

    if args.config:
        config_dict = config.get_args(args.config)
        future_steps = 1
        train()
    else:
        raise ValueError("Config file not set. Use '--config <path_to_file>' to load a configuration.")