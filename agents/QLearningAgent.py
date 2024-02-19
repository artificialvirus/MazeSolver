# This file contains the QLearningAgent class, which is an implementation of the Q-learning algorithm for reinforcement learning.
# File: agents/QLearningAgent.py

import numpy as np

class QLearningAgent:
    def __init__(self, num_states, num_actions, alpha=0.02, gamma=0.8, epsilon=0.5):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q_table = {}  # Use a dictionary to handle dynamic state-action space

    def choose_action(self, state, test=False):
        state_key = (state[0], state[1])
        if test:
            # Choose the best action based on the learned Q-values (exploitation only)
            action = np.argmax(self.q_table.get(state_key, np.zeros(self.num_actions)))
        else:
            if np.random.uniform(0, 1) < self.epsilon:
                # Exploration
                action = np.random.choice(self.num_actions)
            else:
                # Exploitation
                action = np.argmax(self.q_table.get(state_key, np.zeros(self.num_actions)))
        return action

    def update(self, state, action, reward, next_state):
        state_key = (state[0], state[1])
        next_state_key = (next_state[0], next_state[1])
        # Initialize state-action space if not exist
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.num_actions)
        # Q-learning update
        q_old = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] = q_old + self.alpha * (reward + self.gamma * max_next_q - q_old)
