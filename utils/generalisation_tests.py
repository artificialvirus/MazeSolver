# This file contains the code to run generalization tests for the agents
# File: utils/generalisation_tests.py

import numpy as np
from environment.mazes.enhanced_environment import EnhancedEnvironment
from agents.QNetwork import QNetwork
from agents.DQNAgent import DQNAgent
from agents.QLearningAgent import QLearningAgent
from agents.SarsaAgent import SarsaAgent
from evaluation.EvaluatePerformance import run_episode, test_agent

# maze size
rows, cols = 25, 25

# Training and testing parameters
num_episodes = 5
num_tests = 1

# Creating an environment and the agents
env = EnhancedEnvironment(rows, cols)
testing_env = EnhancedEnvironment(rows, cols)
state_size = 6  
action_size = 4  
agents = {
    "Q-learning": QLearningAgent(state_size, action_size),
    "Sarsa": SarsaAgent(state_size, action_size),
    "DQN": DQNAgent(state_size, action_size)

}

def create_generalization_test_environment(width, height, goal_position):
    environment = EnhancedEnvironment(width, height)
    # Adjust generate_environment method to accommodate dynamic goal positions
    environment.goal = goal_position
    environment.generate_environment()  # Make sure this method correctly sets the goal
    return environment

def run_generalization_tests(agent, test_mazes, num_tests_per_maze=10):
    generalization_scores = []
    for maze in test_mazes:
        test_env = create_generalization_test_environment(maze['width'], maze['height'], maze['goal'])
        test_env.maze = maze['layout']  # Assuming each maze dict has 'layout', 'width', 'height', and 'goal'
        avg_reward = test_agent(agent, test_env, num_tests=num_tests_per_maze)
        generalization_scores.append(avg_reward)
    overall_avg = np.mean(generalization_scores)
    print(f"Generalization Test Average Reward: {overall_avg}")
    return overall_avg

test_mazes = [
    {'layout': np.array([...]), 'width': 25, 'height': 25, 'goal': (24, 24)},
    # Add more mazes with different layouts and goals
]

for agent_name, agent in agents.items():
    print(f"Running Generalization Tests for {agent_name}")
    run_generalization_tests(agent, test_mazes)
