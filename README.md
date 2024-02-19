<<<<<<< HEAD
# MazeSolver
=======
Reinforcement Learning in Gridworld
This file contains code that illustrates the implementation of three different Reinforcement Learning (RL) algorithms on a grid-based environment:

Q-Learning
Deep Q-Network (DQN)
Sarsa
These RL algorithms are implemented from scratch for the purpose of learning and comparison.

Overview
The task in this code is to guide an agent in a grid-based environment from a starting point to the goal point while avoiding obstacles. The grid is randomly generated as a maze where the agent can move in four directions: up, down, left, and right. Each step the agent takes gives a small negative reward, hitting a wall gives a larger negative reward, and reaching the goal gives a positive reward.

Dependencies
Python 3.8 or later
PyTorch
NumPy
matplotlib
seaborn
Pygame
Getting Started
Install the required packages: pip install -r requirements.txt
Run the main script: python RL-maze.py
Implementation Details
The RL-maze.py file contains the primary code for running the simulations. It initializes the grid-based environment, the agents, and sets the parameters for the learning algorithms.

The GridEnvironment class defines the environment. It includes methods for stepping through the environment, resetting it, and rendering the environment on the screen using Pygame.

Each agent is defined in its own class: QLearningAgent, DQNAgent, and SarsaAgent. These classes include methods for choosing actions (based on the current policy) and updating the value function based on received rewards.

The generate_maze function creates a random grid-based environment. The grid is saved to a file, which can be loaded for future runs to ensure consistency.

At the end of the script, the rewards received during training are plotted to compare the performance of the different agents.

Notes
The parameters for the learning algorithms and the environment (like size of the grid, learning rates, discount factor, etc.) can be adjusted in the RL-maze.py script.

The agents' performance can vary based on the complexity of the grid (maze), the size of the grid, and the values of the learning parameters.

Note that the project involves a lot of randomness and results may vary between runs.

Alper Onder - sc19ao (Implementation Part)


>>>>>>> 2c45028 (Initial commit.)
