# This file contains the QNetwork class, which is an implementation of the Q-network for reinforcement learning.
# File: agents/QNetwork.py

import torch
import torch.nn as nn

#Use a more sophisticated model
# CNN model (considering the maze as an image)
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)
