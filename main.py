import argparse
from multiprocessing import freeze_support
import torch
from model.brain_behaviour_classifier import BrainBehaviourClassifier
from submodules.TimeSeriesDL.utils.config import config
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import optuna

from torch.utils.data import DataLoader
from data.dataset import BrainDataset

from data.utils import get_dataset_matrix, get_meshes

# some general brain information and proposed models: https://arxiv.org/pdf/2201.04229.pdf

config.register_model("BrainBehaviourClassifier", BrainBehaviourClassifier)

config_dict = None


def train():
    # define parameters (depending on device, lower the precision to save memory)
    device = config_dict["device"]
    precision = torch.float16 if device == "cuda" else torch.float32

    # load the data, normalize them and convert them to tensor
    dataset = BrainDataset(**config_dict["dataset_args"])
    split_sizes = [int(math.ceil(len(dataset) * 0.8)), int(math.floor(len(dataset) * 0.2))]

    trainset, valset = torch.utils.data.random_split(dataset, split_sizes)
    trainloader = DataLoader(trainset, **config_dict["dataloader_args"])
    valloader = DataLoader(valset, **config_dict["dataloader_args"])

    # load test dataset
    config_dict["dataset_args"]["d_type"] = "test"
    config_dict["dataset_args"]["noise"] = False
    testset = BrainDataset(**config_dict["dataset_args"])
    testloader = DataLoader(testset, **config_dict["dataloader_args"])

    # create the model
    model = BrainBehaviourClassifier(**config_dict["model_args"])
    model.use_device(device)

    config_dict["evaluation"] = model.log_path
    config.store_args(f"{model.log_path}/config.yml", config_dict)

    # train the model
    model.learn(loader=trainloader, validate=valloader, test=testloader, epochs=config_dict["train_epochs"])
    model.save_to_default()


def testing():
    # create a dummy input
    # batch_size, sequence length, sample (248 values reshaped to 8x31)
    x = torch.zeros((64, 4, 8, 31))

    model = BrainBehaviourClassifier()
    out = model(x)


def visualize_data():
    file = "./data/data/Intra/train/rest_105923_1.h5"
    matrix = get_dataset_matrix(file)

    # TODO: normalization + downsampling + sequencing

    time_steps = 50
    # Get 2D meshes for N time steps
    meshes = get_meshes(matrix, time_steps)
    from matplotlib.animation import FuncAnimation
    fig, ax = plt.subplots()

    def update(i):
        H = meshes[:, :, i]
        ax.imshow(H, interpolation='none')
        ax.set_axis_off()

    anim = FuncAnimation(fig, update, frames=time_steps, interval=200)
    plt.show()


l = list(range(50, 300))
new_l = [item for item in l if item % 10 == 0]
trial_seq_lens = []


def seq_len_objective(trial):
    x = new_l[trial.number]

    config_dict["dataset_args"]["sequence_length"] = x
    trial_seq_lens.append(x)
    print("Training model with sequence length: " + str(x))
    model = train()

    (acc, var, mse, rmse, mae) = model.test_stats
    return acc


batch_sizes = [2 ** x for x in range(5, 11)]
trail_batch_sizes = []


def batch_size_objective(trail):
    x = batch_sizes[trail.number]
    config_dict["dataloader_args"]["batch_size"] = x
    trail_batch_sizes.append(x)
    print("Training model with batch size: " + str(x))
    model = train()

    (acc, var, mse, rmse, mae) = model.test_stats
    return acc


downsample_bys = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
trail_downsample_bys = []


def downsampling_objective(trail):
    x = downsample_bys[trail.number]
    config_dict["dataset_args"]["downsample_by"] = x
    trail_downsample_bys.append(x)
    print("Training model downsampling of: " + str(x))
    model = train()

    (acc, var, mse, rmse, mae) = model.test_stats
    return acc


def fine_tune_seq_len():
    study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
    study.optimize(seq_len_objective, n_trials=len(new_l))

    best_seq_len = trial_seq_lens[study.best_trial.number]
    print("Best performing seqence length is " + str(best_seq_len))


def fine_tune_batch_siez():
    study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
    study.optimize(batch_size_objective, n_trials=len(batch_sizes))

    best_batch_size = trial_seq_lens[study.best_trial.number]
    print("Best performing batch size is " + str(best_batch_size))


def fine_tune_downsampling():
    study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
    study.optimize(batch_size_objective, n_trials=len(downsample_bys))

    best_downsample_rate = trial_seq_lens[study.best_trial.number]
    print("Best performing batch size is " + str(best_downsample_rate))


if __name__ == "__main__":
    freeze_support()
    parser = argparse.ArgumentParser(description="This program trains and tests a deep " +
                                                 "learning model to detect a behaviour based on brain data")
    parser.add_argument("--config", dest="config", help="Set path to config file.")
    parser.add_argument("--debug", dest="debug", help="Runs the model with a dummy input to debug it.",
                        action="store_true")
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
