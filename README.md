# Reinforcement Learning Framework
This project provides a comprehensive framework for experimenting with different reinforcement learning (RL) algorithms in maze-solving tasks. It includes implementations of Q-learning, Sarsa, and Deep Q-Network (DQN) agents, a customizable maze environment, and tools for evaluating and visualizing agent performance.

## Features
### Reinforcement Learning Agents: 
Implementations of Q-learning, Sarsa, and DQN agents.
### Enhanced Environment: 
A maze environment that can be easily customized for various levels of complexity.
### Performance Evaluation: 
Tools for evaluating agent performance through rewards, steps, and learning speed metrics.
### Visualization: 
Utilities for visualizing the learning process, maze navigation, and performance metrics.
### Hyperparameter Optimization: 
Framework for optimizing the agents' hyperparameters using Bayesian optimization.

## Getting Started
### Prerequisites
Python 3.8+
PyTorch
NumPy
Matplotlib
Seaborn
Pygame
scikit-optimize (skopt)
Installation
### Clone the repository:
```
git clone https://github.com/artificialvirus/MazeSolver.git
```
### Install the required Python packages:
```
pip install torch numpy matplotlib seaborn pygame scikit-optimize
```
## Running the Agents
### Navigate to the project directory and run the main script:
```
python main.py
```
This will start a training and testing cycle for each implemented RL agent, displaying their performance and visualizations of their behavior in the maze.

## Hyperparameter Optimization
### To run hyperparameter optimization, uncomment the following line in main.py:
```
# run_hyperparameter_optimization(agents, env, num_episodes)
```
and then run the script as described above.

## Testing
To run the unit tests, navigate to the project directory and execute:
```
python -m unittest
```
This will run tests to verify the correct behavior of the agents, environment, and utilities.

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues to improve the project.

## License
This project is licensed under the MIT License - see the LICENSE file for details.