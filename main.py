import argparse
from model.brain_behaviour_classifier import BrainBehaviourClassifier
from submodules.TimeSeriesDL.utils.config import config

config.register_model("BrainBehaviourClassifier", BrainBehaviourClassifier)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="This program trains and tests a deep " + 
                                     "learning model to detect a behaviour based on brain data")
    parser.add_argument("--config", dest="config", help="Set path to config file.")
    args = parser.parse_args()

    if args.config:
        pass
    else:
        raise argparse.ArgumentError("Config file not set. Use '--config <path_to_file>' to load a configuration.")