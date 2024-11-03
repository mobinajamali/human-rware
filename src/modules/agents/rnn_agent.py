# code adapted from https://github.com/wendelinboehmer/dcg

import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, use_rnn):
        super(RNNAgent, self).__init__()
        self.hidden_dim = 128
        self.n_actions = 5
        self.use_rnn = use_rnn

        self.fc1 = nn.Linear(input_shape, self.hidden_dim)
        if self.use_rnn:
            self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        else:
            self.rnn = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        if self.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h

