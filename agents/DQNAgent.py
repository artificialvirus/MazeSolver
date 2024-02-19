# This file contains the DQNAgent class, which is an implementation of the DQN algorithm for reinforcement learning.
# File: agents/DQNAgent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from agents.QNetwork import QNetwork  

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lr=0.001, gamma=0.85, epsilon=0.5, buffer_size=10000, batch_size=64, update_target_every=5):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.learn_step_counter = 0  # For updating the target network

        self.memory = ReplayBuffer(buffer_size)
        self.model = QNetwork(state_size, action_size, hidden_size)
        self.target_model = QNetwork(state_size, action_size, hidden_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.loss_history = []  # Stores loss for each update
        self.avg_q_value_history = []  # Stores average Q-value for each episode

    def choose_action(self, state, test=False):
        if test or np.random.rand() > self.epsilon:  # No exploration if testing
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
            return q_values.argmax().item()
        else:
            return random.randrange(self.action_size)


    def update(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        curr_q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Track average Q-values
        avg_q_value = curr_q_values.mean().item()
        self.avg_q_value_history.append(avg_q_value)

        loss = self.criterion(curr_q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store loss
        self.loss_history.append(loss.item())

        if self.learn_step_counter % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        self.learn_step_counter += 1

    def report_performance_metrics(self):
        # Calculate and return average loss and Q-value for the current episode
        avg_loss = sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0
        avg_q_value = sum(self.avg_q_value_history) / len(self.avg_q_value_history) if self.avg_q_value_history else 0
        self.loss_history.clear()  # Reset for the next episode
        self.avg_q_value_history.clear()  # Reset for the next episode
        return avg_loss, avg_q_value
