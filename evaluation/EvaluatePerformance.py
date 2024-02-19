# This file contains the code to run an episode and test the agent's performance
# File: evaluation/evaluate_performance.py

import numpy as np
import pygame
from environment.mazes.enhanced_environment import EnhancedEnvironment
from agents.QNetwork import QNetwork
from agents.DQNAgent import DQNAgent
from agents.QLearningAgent import QLearningAgent
from agents.SarsaAgent import SarsaAgent


def run_episode(agent, environment):
    state = environment.reset()  # Reset the environment at the start of each episode
    total_reward = 0
    num_steps = 0  # Initialize step counter
    done = False

    while not done:
        action = agent.choose_action(state, test=False)  # Choose an action
        next_state, reward, done, _ = environment.step(action)  # Environment returns next state and reward

        # For Sarsa, determine the next action here to pass it to the update method
        if isinstance(agent, SarsaAgent):
            next_action = agent.choose_action(next_state, test=False)  # Choose next action for Sarsa
            agent.update(state, action, reward, next_state, next_action)  # Update for Sarsa includes next_action

        elif isinstance(agent, DQNAgent):
            agent.update(state, action, reward, next_state, done)

        else:
            # For other agents, use the regular update method
            agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward  # Accumulate the reward
        num_steps += 1  # Increment step counter

        environment.render()  # Optional: Render the environment to visualize the agent's behavior

        for event in pygame.event.get():  # Allow manual closing of the simulation window
            if event.type == pygame.QUIT:
                done = True

    return total_reward, num_steps

def test_agent(agent, environment, num_tests=1):
    agent.epsilon = 0  # Ensure no exploration during testing
    total_rewards = []
    for _ in range(num_tests):
        state = environment.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state, test=True)  # Ensure exploration is off during testing
            next_state, reward, done, _ = environment.step(action)

            # For Sarsa, determine the next action here to pass it to the update method
            if isinstance(agent, SarsaAgent):
                next_action = agent.choose_action(next_state, test=False)  # Choose next action for Sarsa
                agent.update(state, action, reward, next_state, next_action)  # Update for Sarsa includes next_action

            elif isinstance(agent, DQNAgent):
                agent.update(state, action, reward, next_state, done)

            else:
                # For other agents, use the regular update method
                agent.update(state, action, reward, next_state)
                
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward during Testing: {avg_reward}")
    return avg_reward
