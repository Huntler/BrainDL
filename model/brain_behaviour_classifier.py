from datetime import datetime
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter
import torch
from submodules.TimeSeriesDL.model.base_model import BaseModel


# definitely a CNN again, maybe also LSTM
# reshape the input to (8, 31) and apply small filter (3x3 or 5x5)


# TODO: make values variable and define them in the classes constructor


class BrainBehaviourClassifier(BaseModel):
    def __init__(self) -> None:
        # set up tensorboard
        self.__tb_sub = datetime.now().strftime("%m-%d-%Y_%H%M%S")
        self._tb_path = f"runs/BrainBehaviourClassifier/{self.__tb_sub}"
        self._writer = SummaryWriter(self._tb_path)

        super(BrainBehaviourClassifier, self).__init__()

        # first part of the neural network is CNN only which tries to predict
        # one of the 4 classes without taking a sequence into account
        self.__conv_1 = torch.nn.Conv2d(1, 32, 3, 1, 0)
        self.__conv_1_bn = torch.nn.BatchNorm2d(32)

        self.__conv_2 = torch.nn.Conv2d(32, 64, 3, 1, 0)
        self.__conv_2_bn = torch.nn.BatchNorm2d(64)

        self.__conv_3 = torch.nn.Conv1d(64, 128, 4, 1, 0)
        self.__conv_3_bn = torch.nn.BatchNorm1d(128)

        self.__linear_1 = torch.nn.Linear(128, 64)
        self.__linear_2 = torch.nn.Linear(64, 4)

        self.__l_1 = torch.nn.CrossEntropyLoss()

        # second part of the neural network is a LSTM which takes the previous
        # output as an input and tries to predict one of the 4 classes with
        # taking the sequence into account
        self.__lstm = torch.nn.LSTM(4, 32, batch_first=True)
        self.__linear_3 = torch.nn.Linear(32, 4)

        self.__l_2 = torch.nn.CrossEntropyLoss()
    
    def forward(self, x: torch.tensor) -> Tuple[torch.tensor]:
        # forward passing into the CNN build of 3 Conv layers and 2 
        # max pooling layers
        new_x = torch.empty((64, 4, 32, 3, 14))
        for i, x_t in enumerate(x.split(1, dim=1)):
            x_t = self.__conv_1(x_t)
            x_t = self.__conv_1_bn(x_t)
            x_t = torch.relu(x_t)
            x_t = torch.max_pool2d(x_t, 2)
            new_x[:, i] = x_t

        x = new_x

        new_x = torch.empty((64, 4, 64, 1, 4))
        for i, x_t in enumerate(x.split(1, dim=1)):
            x_t = torch.flatten(x_t, 0, 1)
            x_t = self.__conv_2(x_t)
            x_t = self.__conv_2_bn(x_t)
            x_t = torch.relu(x_t)
            x_t = torch.max_pool2d(x_t, (1, 3))
            new_x[:, i] = x_t

        x = torch.flatten(new_x, -2, -1)
        new_x = torch.empty((64, 4, 128, 1))
        for i, x_t in enumerate(x.split(1, dim=1)):
            x_t = torch.flatten(x_t, 0, 1)
            x_t = self.__conv_3(x_t)
            x_t = self.__conv_3_bn(x_t)
            x_t = torch.relu(x_t)
            new_x[:, i] = x_t

        x = torch.flatten(new_x, -2, -1)
        x = self.__linear_1(x)
        x = torch.relu(x)
        x = self.__linear_2(x)
        cnn_out = torch.relu(x)
        
        # forward passing the CNN output into the LSTM and try to classify the
        # whole sequence
        hidden = (torch.rand(1, 64, 32), torch.rand(1, 64, 32))
        x, hidden = self.__lstm(cnn_out, hidden)
        # x = x[:, -1]
        # x = torch.unsqueeze(x, 1)

        x = self.__linear_3(x)
        lstm_out = torch.relu(x)

        # returns output of CNN and LSTM
        return cnn_out, lstm_out