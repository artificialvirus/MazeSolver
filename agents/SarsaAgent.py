# This file contains the SarsaAgent class, which is an implementation of the Sarsa algorithm for reinforcement learning.
# File: agents/SarsaAgent.py

import numpy as np

class SarsaAgent:
    def __init__(self, num_states, num_actions, alpha=0.02, gamma=0.80, epsilon=0.5):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # Use a dictionary to handle dynamic state-action space

    def choose_action(self, state, test=False):
        state_key = (state[0], state[1])
        # Ensure there is an entry for the current state in Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)

        if test:
            # Exploitation only for testing, no random choice.
            action = np.argmax(self.q_table[state_key])
        else:
            # Epsilon-greedy action selection for training
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.num_actions)  # Exploration
            else:
                action = np.argmax(self.q_table[state_key])  # Exploitation
        return action
    

    def update(self, state, action, reward, next_state, next_action):
        state_key = (state[0], state[1])
        next_state_key = (next_state[0], next_state[1])
        # Initialize state-action space if not exist
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.num_actions)
        # Sarsa update
        q_old = self.q_table[state_key][action]
        next_q = self.q_table[next_state_key][next_action]
        self.q_table[state_key][action] = q_old + self.alpha * (reward + self.gamma * next_q - q_old)
