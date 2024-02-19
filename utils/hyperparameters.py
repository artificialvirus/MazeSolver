# This file contains the code for hyperparameter optimization using Bayesian optimization.
# File: utils/hyperparameters.py

import numpy as np
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from environment.mazes.enhanced_environment import EnhancedEnvironment
from agents.SarsaAgent import SarsaAgent
from agents.DQNAgent import DQNAgent
from agents.QLearningAgent import QLearningAgent
from evaluation.EvaluatePerformance import run_episode

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

# Define the space of hyperparameters to search
space_dqn = [
    Real(1e-5, 1e-1, name='lr', prior='log-uniform'),
    Real(0.8, 0.99, name='gamma'),
    Real(0.1, 0.9, name='epsilon')
]
space = [
    Real(1e-5, 1e-1, name='alpha', prior='log-uniform'),
    Real(0.8, 0.99, name='gamma'),
    Real(0.1, 0.9, name='epsilon')
]

# Define the objective function to minimize
@use_named_args(space)
def objective_qlearning(**params):
    env = EnhancedEnvironment(rows, cols)  # Assuming rows and cols are defined
    agent = QLearningAgent(num_states=6, num_actions=4, **params)
    total_rewards = []
    for _ in range(num_episodes):
        total_reward, _ = run_episode(agent, env)
        total_rewards.append(total_reward)
        print(total_reward)
        
    return -np.mean(total_rewards)  # Return negative average reward

@use_named_args(space)
def objective_sarsa(**params):
    env = EnhancedEnvironment(rows, cols)  # Assuming rows and cols are defined
    agent = SarsaAgent(num_states=6, num_actions=4, **params)
    total_rewards = []
    for _ in range(num_episodes):
        total_reward, _ = run_episode(agent, env)
        total_rewards.append(total_reward)
        print(total_reward)
        
    return -np.mean(total_rewards)  # Return negative average reward

@use_named_args(space_dqn)
def objective_dqn(**params):
    # Initialize the environment
    env = EnhancedEnvironment(rows, cols)
    # Create an instance of DQNAgent with the given hyperparameters
    agent = DQNAgent(state_size=6, action_size=4, hidden_size=64, **params)
    
    total_rewards = []
    # Run a predefined number of episodes to evaluate the agent
    for _ in range(num_episodes):  # num_episodes is defined globally
        total_reward, _ = run_episode(agent, env)
        total_rewards.append(total_reward)
        print(total_reward)
    
    # Objective is to maximize the total reward, hence minimize negative reward
    return -np.mean(total_rewards)


# # Execute Bayesian optimization to find the best hyperparameters
# result_qlearning = gp_minimize(objective_qlearning, space, n_calls=20, random_state=0)
# print("Best parameters for QLearningAgent: {}".format(result_qlearning.x))
# print("Best average reward: {}".format(-result_qlearning.fun))

# result_sarsa = gp_minimize(objective_sarsa, space, n_calls=20, random_state=0)
# print("Best parameters for SarsaAgent: {}".format(result_sarsa.x))
# print("Best average reward: {}".format(-result_sarsa.fun))

# result_dqn = gp_minimize(objective_dqn, space_dqn, n_calls=10, random_state=0)
# print("Best parameters: {}".format(result_dqn.x))
# print("Best average reward: {}".format(-result_dqn.fun))

# # Save the results
# # Save to performance data folder which is inside data directory the results of the hyperparameter optimization should be in data/performance_data
# np.save('data/performance_data/qlearning_hyperparameters.npy', result_qlearning)
# np.save('data/performance_data/sarsa_hyperparameters.npy', result_sarsa)
# np.save('data/performance_data/dqn_hyperparameters.npy', result_dqn)

# # Save the best hyperparameters to a file
# with open('data/performance_data/best_hyperparameters.txt', 'w') as f:
#     f.write("Best parameters for QLearningAgent: {}\n".format(result_qlearning.x))
#     f.write("Best average reward: {}\n".format(-result_qlearning.fun))
#     f.write("Best parameters for SarsaAgent: {}\n".format(result_sarsa.x))
#     f.write("Best average reward: {}\n".format(-result_sarsa.fun))
#     f.write("Best parameters for DQNAgent: {}\n".format(result_dqn.x))
#     f.write("Best average reward: {}\n".format(-result_dqn.fun))
    
# Write a function that can be used in main.py to run hyperparameter optimization
def run_hyperparameter_optimization(agents, env, num_episodes):
    # Execute Bayesian optimization to find the best hyperparameters
    
    print("Running hyperparameter optimization...")
    print("This may take a while...")

    print("Hyperparameter optimization for QLearningAgent...")
    result_qlearning = gp_minimize(objective_qlearning, space, n_calls=20, random_state=0)
    print("Best parameters for QLearningAgent: {}".format(result_qlearning.x))
    print("Best average reward: {}".format(-result_qlearning.fun))

    print("Hyperparameter optimization for SarsaAgent...")
    result_sarsa = gp_minimize(objective_sarsa, space, n_calls=20, random_state=0)
    print("Best parameters for SarsaAgent: {}".format(result_sarsa.x))
    print("Best average reward: {}".format(-result_sarsa.fun))

    print("Hyperparameter optimization for DQNAgent...")
    result_dqn = gp_minimize(objective_dqn, space_dqn, n_calls=10, random_state=0)
    print("Best parameters: {}".format(result_dqn.x))
    print("Best average reward: {}".format(-result_dqn.fun))

    # Save the results
    # Save to performance data folder which is inside data directory the results of the hyperparameter optimization should be in data/performance_data
    np.save('data/performance_data/qlearning_hyperparameters.npy', result_qlearning)
    np.save('data/performance_data/sarsa_hyperparameters.npy', result_sarsa)
    np.save('data/performance_data/dqn_hyperparameters.npy', result_dqn)

    # Save the best hyperparameters to a file
    with open('data/performance_data/best_hyperparameters.txt', 'w') as f:
        f.write("Best parameters for QLearningAgent: {}\n".format(result_qlearning.x))
        f.write("Best average reward: {}\n".format(-result_qlearning.fun))
        f.write("Best parameters for SarsaAgent: {}\n".format(result_sarsa.x))
        f.write("Best average reward: {}\n".format(-result_sarsa.fun))
        f.write("Best parameters for DQNAgent: {}\n".format(result_dqn.x))
        f.write("Best average reward: {}\n".format(-result_dqn.fun))
    
    return result_qlearning, result_sarsa, result_dqn
