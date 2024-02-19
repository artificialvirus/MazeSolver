# This file contains functions to save and load the environment and the agent
# File: utils/save_load.py

import pickle
import torch
from environment.mazes.enhanced_environment import EnhancedEnvironment

def save_environment(environment, filename):
    with open(filename, 'wb') as f:
        pickle.dump({
            'maze': environment.maze,
            'state': environment.state,
            'goal': environment.goal,
            'width': environment.width,
            'height': environment.height,
        }, f)

def load_environment(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    environment = EnhancedEnvironment(data['width'], data['height'])
    environment.maze = data['maze']
    environment.state = data['state']
    environment.goal = data['goal']
    return environment

def save_agent(agent, filename):
    torch.save(agent.model.state_dict(), filename)

def load_agent(agent, filename):
    agent.model.load_state_dict(torch.load(filename))
    agent.model.eval()  # Set the network to evaluation mode
