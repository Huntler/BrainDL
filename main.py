import argparse

import torch
from model.brain_behaviour_classifier import BrainBehaviourClassifier
from submodules.TimeSeriesDL.utils.config import config

# some general brain information and proposed models: https://arxiv.org/pdf/2201.04229.pdf

config.register_model("BrainBehaviourClassifier", BrainBehaviourClassifier)


def testing():
    # create a dummy input
    # batch_size, sequence length, sample (248 values reshaped to 8x31)
    x = torch.zeros((64, 4, 8, 31))

    model = BrainBehaviourClassifier()
    out = model(x)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="This program trains and tests a deep " + 
                                     "learning model to detect a behaviour based on brain data")
    parser.add_argument("--config", dest="config", help="Set path to config file.")
    parser.add_argument("--debug", dest="debug", help="Runs the model with a dummy input to debug it.", action="store_true")
    args = parser.parse_args()

    if args.debug:
        testing()
        quit()

    if args.config:
        pass
    else:
        raise ValueError("Config file not set. Use '--config <path_to_file>' to load a configuration.")