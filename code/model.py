import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: make the model work on mps

class Model(nn.Module):
    def __init__(self, input_size=16, hidden_size=50, num_layers=2, dropout=0.2, device='cpu'):
        super(Model, self).__init__()
        self.input_size = input_size  # the number of expected features in the input x
        self.hidden_size = hidden_size  # the number of features in the hidden state h
        self.num_layers = num_layers  # the number of recurrent layers
        self.dropout = dropout  # if non-zero, introduces a dropout layer on the outputs of each LSTM layer except the last layer
        self.device = device
        self.accuracy = 0
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=(False if device == 'mps' else True), dropout=dropout)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout).to(self.device)
        self.fc = nn.Linear(hidden_size, 1).to(self.device)

    def forward(self, x):
        # x is a 3d tensor of shape (batch_size, sequence_length, input_size)
        # get the initial hidden and cell state for the first input
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # get the output and final hidden and cell state
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        # get the final output
        out = torch.sigmoid(self.fc(out[:, -1, :]))
        return out
