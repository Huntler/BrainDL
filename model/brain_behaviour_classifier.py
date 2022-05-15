from datetime import datetime
from typing import List, Tuple
from unicodedata import bidirectional
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from submodules.TimeSeriesDL.model.base_model import BaseModel
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm


# definitely a CNN again, maybe also LSTM
# reshape the input to (8, 31) and apply small filter (3x3 or 5x5)


# TODO: make values variable and define them in the classes constructor


class BrainBehaviourClassifier(BaseModel):
    def __init__(self, lr: float = 1e-3, lr_decay: float = 9e-1, adam_betas: List[float] = [9e-1, 999e-3]) -> None:
        # set up tensorboard
        self.__tb_sub = datetime.now().strftime("%m-%d-%Y_%H%M%S")
        self._tb_path = f"runs/BrainBehaviourClassifier/{self.__tb_sub}"
        self._writer = SummaryWriter(self._tb_path)

        super(BrainBehaviourClassifier, self).__init__()

        # first part of the neural network is CNN only which tries to predict
        # one of the 4 classes without taking a sequence into account
        self.__conv_1 = torch.nn.Conv2d(1, 32, 5, 2, 0)
        self.__conv_1_bn = torch.nn.BatchNorm2d(32, track_running_stats=False)

        self.__conv_2 = torch.nn.Conv2d(32, 64, 5, 2, 0)
        self.__conv_2_bn = torch.nn.BatchNorm2d(64, track_running_stats=False)

        self.__conv_3 = torch.nn.Conv1d(64, 128, 6, 1, 0)
        self.__conv_3_bn = torch.nn.BatchNorm1d(128, track_running_stats=False)

        self.__linear_1 = torch.nn.Linear(128, 64)
        self.__linear_2 = torch.nn.Linear(64, 4)

        # second part of the neural network is a LSTM which takes the previous
        # output as an input and tries to predict one of the 4 classes with
        # taking the sequence into account
        self.__lstm = torch.nn.LSTM(4, 32, num_layers=8, dropout=0.2, bidirectional=False, batch_first=True)
        self.__linear_3 = torch.nn.Linear(32, 4)

        self.__loss_func = torch.nn.CrossEntropyLoss()
        self._optim = torch.optim.AdamW(self.parameters(), lr=lr, betas=adam_betas)
        self._scheduler = ExponentialLR(self._optim, gamma=lr_decay)
        self.__sample_position = 0

    def learn(self, loader: DataLoader, epochs: int = 1) -> None:
        dev_name = self._device_name if self._device == "cuda" else "CPU"
        print(f"Starting training on {dev_name}")

        self.train()
        for epoch in tqdm(range(epochs)):
            for x, y in loader:
                x = x.to(self._device)
                y = y.to(self._device)

                self._optim.zero_grad()
                _y = self(x)
                y = torch.flatten(y)
                loss = self.__loss_func(_y, y)

                loss.backward()
                self._optim.step()

                self._writer.add_scalar(
                    "Train/loss", loss, self.__sample_position)

                self.__sample_position += x.size(0)

            # if there is an adaptive learning rate (scheduler) available
            if self._scheduler:
                self._scheduler.step()
                lr = self._scheduler.get_last_lr()[0]
                self._writer.add_scalar("Train/learning_rate", lr, epoch)


    def validate(self, loader: DataLoader) -> None:
        losses = []
        for x, y in loader:
            x = x.to(self._device)
            y = y.to(self._device)

            _y = self(x)
            y = torch.flatten(y)
            loss = self.__loss_func(_y, y)
            losses.append(loss.detach().cpu().item())

        losses = np.array(losses)
        self._writer.add_scalar(
            "Validation/loss", np.mean(losses), self.__sample_position)


    def accuracy(self, loader: DataLoader) -> float:
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self._device)
                y = y.to(self._device)
                _y = self(x)

                total += y.size(0)
                correct += (_y == y).sum().item()

        return correct * 100 // total
    
    def forward(self, x: torch.tensor) -> Tuple[torch.tensor]:
        batch_size, seq_size, dim_1, dim_2 = x.shape

        # forward passing into the CNN build of 3 Conv layers
        new_x = torch.empty((batch_size, seq_size, 32, 8, 9), device=self._device)
        for i, x_t in enumerate(x.split(1, dim=1)):
            x_t = self.__conv_1(x_t)
            #x_t = self.__conv_1_bn(x_t)
            x_t = torch.relu(x_t)
            new_x[:, i] = x_t

        x = new_x

        new_x = torch.empty((batch_size, seq_size, 64, 2, 3), device=self._device)
        for i, x_t in enumerate(x.split(1, dim=1)):
            x_t = torch.flatten(x_t, 0, 1)
            x_t = self.__conv_2(x_t)
            #x_t = self.__conv_2_bn(x_t)
            x_t = torch.relu(x_t)
            new_x[:, i] = x_t

        x = torch.flatten(new_x, -2, -1)
        new_x = torch.empty((batch_size, seq_size, 128, 1), device=self._device)
        for i, x_t in enumerate(x.split(1, dim=1)):
            x_t = torch.flatten(x_t, 0, 1)
            x_t = self.__conv_3(x_t)
            #x_t = self.__conv_3_bn(x_t)
            x_t = torch.relu(x_t)
            new_x[:, i] = x_t

        x = torch.flatten(new_x, -2, -1)
        x = self.__linear_1(x)
        x = torch.relu(x)
        x = self.__linear_2(x)
        cnn_out = torch.softmax(x, dim=-1)
        
        # forward passing the CNN output into the LSTM and try to classify the
        # whole sequence
        hidden = (torch.zeros(8, batch_size, 32, device=self._device), 
                  torch.zeros(8, batch_size, 32, device=self._device))
        x, hidden = self.__lstm(cnn_out)
        x = x[:, -1]
        # x = torch.unsqueeze(x, 1)

        x = self.__linear_3(x)
        lstm_out = torch.softmax(x, dim=-1)

        # returns output of CNN and LSTM
        return lstm_out