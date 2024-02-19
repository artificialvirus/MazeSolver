# This file contains test cases for the QLearningAgent, SarsaAgent, QNetwork, save_load, EvaluatePerformance, generalisation_tests and main files.
# File: tests/test.py

import numpy as np
import os
import torch
import unittest
from agents.DQNAgent import DQNAgent
from agents.QLearningAgent import QLearningAgent
from agents.SarsaAgent import SarsaAgent
from agents.QNetwork import QNetwork
from environment.mazes.enhanced_environment import EnhancedEnvironment
from evaluation.EvaluatePerformance import run_episode, test_agent
from utils.generalisation_tests import run_generalisation_tests
from utils.save_load import save_environment, load_environment, save_agent, load_agent
from utils.hyperparameters import run_hyperparameter_optimization
from utils.visualisation import plot_rewards, plot_steps
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        self.env = EnhancedEnvironment(25, 25)
        self.agent = QLearningAgent(6, 4)

    def test_choose_action(self):
        state = (0, 0)
        action = self.agent.choose_action(state)
        self.assertIn(action, [0, 1, 2, 3])

    def test_update(self):
        state = (0, 0)
        action = 0
        reward = 0
        next_state = (0, 1)
        self.agent.update(state, action, reward, next_state)
        self.assertEqual(self.agent.Q[state][action], 0)

    def test_epsilon_greedy(self):
        state = (0, 0)
        action = self.agent.epsilon_greedy(state)
        self.assertIn(action, [0, 1, 2, 3])

class TestSarsaAgent(unittest.TestCase):
    def setUp(self):
        self.env = EnhancedEnvironment(25, 25)
        self.agent = SarsaAgent(6, 4)

    def test_choose_action(self):
        state = (0, 0)
        action = self.agent.choose_action(state)
        self.assertIn(action, [0, 1, 2, 3])

    def test_update(self):
        state = (0, 0)
        action = 0
        reward = 0
        next_state = (0, 1)
        next_action = 1
        self.agent.update(state, action, reward, next_state, next_action)
        self.assertEqual(self.agent.Q[state][action], 0)

    def test_epsilon_greedy(self):
        state = (0, 0)
        action = self.agent.epsilon_greedy(state)
        self.assertIn(action, [0, 1, 2, 3])

class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        self.env = EnhancedEnvironment(25, 25)
        self.agent = DQNAgent(6, 4)

    def test_choose_action(self):
        state = (0, 0)
        action = self.agent.choose_action(state)
        self.assertIn(action, [0, 1, 2, 3])

    def test_update(self):
        state = (0, 0)
        action = 0
        reward = 0
        next_state = (0, 1)
        done = False
        self.agent.update(state, action, reward, next_state, done)
        self.assertEqual(self.agent.Q[state][action], 0)

    def test_epsilon_greedy(self):
        state = (0, 0)
        action = self.agent.epsilon_greedy(state)
        self.assertIn(action, [0, 1, 2, 3])

class TestQNetwork(unittest.TestCase):
    def setUp(self):
        self.env = EnhancedEnvironment(25, 25)
        self.agent = QNetwork(6, 4)

    def test_choose_action(self):
        state = (0, 0)
        action = self.agent.choose_action(state)
        self.assertIn(action, [0, 1, 2, 3])

    def test_update(self):
        state = (0, 0)
        action = 0
        reward = 0
        next_state = (0, 1)
        done = False
        self.agent.update(state, action, reward, next_state, done)
        self.assertEqual(self.agent.Q[state][action], 0)

    def test_epsilon_greedy(self):
        state = (0, 0)
        action = self.agent.epsilon_greedy(state)
        self.assertIn(action, [0, 1, 2, 3])

class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        self.env = EnhancedEnvironment(25, 25)
        self.agent = QLearningAgent(6, 4)

    def test_save_load_environment(self):
        save_environment(self.env, 'test_env')
        loaded_env = load_environment('test_env')
        self.assertEqual(self.env, loaded_env)

    def test_save_load_agent(self):
        save_agent(self.agent, 'test_agent')
        loaded_agent = load_agent('test_agent')
        self.assertEqual(self.agent, loaded_agent)

class TestEvaluatePerformance(unittest.TestCase):
    def setUp(self):
        self.env = EnhancedEnvironment(25, 25)
        self.agent = QLearningAgent(6, 4)

    def test_run_episode(self):
        total_reward, num_steps = run_episode(self.agent, self.env)
        self.assertIsInstance(total_reward, int)
        self.assertIsInstance(num_steps, int)

    def test_test_agent(self):
        test_agent(self.agent, self.env)
        self.assertEqual(self.agent.epsilon, 0)

class TestGeneralisationTests(unittest.TestCase):
    def setUp(self):
        self.env = EnhancedEnvironment(25, 25)
        self.agent = QLearningAgent(6, 4)

    def test_run_generalisation_tests(self):
        run_generalisation_tests(self.agent, self.env)

class TestHyperparameters(unittest.TestCase):
    def setUp(self):
        self.env = EnhancedEnvironment(25, 25)
        self.agents = {
            "Q-learning": QLearningAgent(6, 4),
            "Sarsa": SarsaAgent(6, 4),
            "DQN": DQNAgent(6, 4)
        }

    def test_run_hyperparameter_optimization(self):
        run_hyperparameter_optimization(self.agents, self.env, 5)

class TestMain(unittest.TestCase):
    def setUp(self):
        self.env = EnhancedEnvironment(25, 25)
        self.agents = {
            "Q-learning": QLearningAgent(6, 4),
            "Sarsa": SarsaAgent(6, 4),
            "DQN": DQNAgent(6, 4)
        }

    def test_main(self):
        episode_rewards = {agent_name: [] for agent_name in self.agents}
        episode_steps = {agent_name: [] for agent_name in self.agents}
        learning_speeds = {agent_name: 0 for agent_name in self.agents}
        rewards = {agent_name: [] for agent_name in self.agents}
        satisfactory_performance = 25
        run_hyperparameter_optimization(self.agents, self.env, 5)
        for agent_name, agent in self.agents.items():
            for i_episode in range(5):
                total_reward, num_steps = run_episode(agent, self.env)
                episode_rewards[agent_name].append(total_reward)
                episode_steps[agent_name].append(num_steps)
                if total_reward >= satisfactory_performance and learning_speeds[agent_name] == 0:
                    learning_speeds[agent_name] = i_episode
            for _ in range(1):
                test_agent(agent, self.env)


def main():
    unittest.main()

if __name__ == '__main__':
    main()