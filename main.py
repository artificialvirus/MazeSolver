# Description: This file is the main file for the project. It contains all the necessary imports and configurations for the project.
# File: main.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from agents.DQNAgent import DQNAgent
from agents.QLearningAgent import QLearningAgent
from agents.SarsaAgent import SarsaAgent
from environment.mazes.enhanced_environment import EnhancedEnvironment
from evaluation.EvaluatePerformance import run_episode, test_agent
from utils.hyperparameters import run_hyperparameter_optimization
from utils.save_load import save_environment, load_environment, save_agent, load_agent
from utils.visualisation import plot_rewards, plot_steps

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

# Optionally, run hyperparameter optimization
# run_hyperparameter_optimization(agents, env, num_episodes)

# Training and testing
episode_rewards = {agent_name: [] for agent_name in agents}
episode_steps = {agent_name: [] for agent_name in agents}
learning_speeds = {agent_name: 0 for agent_name in agents}
rewards = {agent_name: [] for agent_name in agents}

satisfactory_performance = 25  # threshold for satisfactory performance

run_hyperparameter_optimization(agents, env, num_episodes)


for agent_name, agent in agents.items():
    print(f"Training and testing {agent_name} agent...")

    # Training
    for i_episode in range(num_episodes):
        total_reward, num_steps = run_episode(agent, env)  # Unpack the total reward and number of steps

        # if isinstance(agent, DQNAgent):
        #     avg_loss, avg_q_value = agent.report_performance_metrics()  # Capture additional metrics after each episode
        #     print(f"Episode {i_episode}: Total Reward = {total_reward}, Total Steps = {num_steps}, Avg Loss = {avg_loss}, Avg Q-Value = {avg_q_value}")

        print(f"Episode {i_episode}: Total Reward = {total_reward}, Total Steps = {num_steps}")
        episode_rewards[agent_name].append(total_reward)
        episode_steps[agent_name].append(num_steps)  # number of steps

        # checking if the agent has reached satisfactory performance
        if total_reward >= satisfactory_performance and learning_speeds[agent_name] == 0:
            learning_speeds[agent_name] = i_episode

    # Testing
    for _ in range(num_tests):
        print(f"Testing {agent_name}")
        test_agent(agent, testing_env)

# To save the current state of environments
save_environment(env, 'current_environment.pkl')
save_environment(testing_env, 'current_testing_environment.pkl')

# To load the environment (use when needed)
env = load_environment('current_environment.pkl')
testing_env = load_environment('current_testing_environment.pkl')



# Plotting
fig, axs = plt.subplots(2)

# Plotting of the rewards during training
for agent_name, reward_list in episode_rewards.items():
    axs[0].plot(reward_list, label=agent_name)
axs[0].set_title('Rewards during training')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Total Reward')
axs[0].legend()

# Bar plot of the average rewards during testing
axs[1].bar(rewards.keys(), rewards.values())
axs[1].set_title('Average rewards during testing')
axs[1].set_xlabel('Agent')
axs[1].set_ylabel('Average Reward')

plt.tight_layout()
plt.show()

reward_df = pd.DataFrame(episode_rewards)
reward_df = reward_df.melt(var_name='Agent', value_name='Total Reward')
plt.figure(figsize=(10, 6))
sns.lineplot(data=reward_df, x=reward_df.index, y='Total Reward', hue='Agent')
plt.title('Total reward per episode for each agent')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

steps_df = pd.DataFrame(episode_steps)
steps_df = steps_df.melt(var_name='Agent', value_name='Total Steps')
plt.figure(figsize=(10, 6))
sns.lineplot(data=steps_df, x=steps_df.index, y='Total Steps', hue='Agent')
plt.title('Total steps per episode for each agent')
plt.xlabel('Episode')
plt.ylabel('Total Steps')
plt.show()

learning_speed_df = pd.Series(learning_speeds).to_frame().reset_index()
learning_speed_df.columns = ['Agent', 'Learning Speed']
plt.figure(figsize=(10, 6))
sns.barplot(data=learning_speed_df, x='Agent', y='Learning Speed')
plt.title('Learning speed for each agent')
plt.xlabel('Agent')
plt.ylabel('Learning Speed')
plt.show()

reward_df = pd.Series(rewards).to_frame().reset_index()
reward_df.columns = ['Agent', 'Final Reward']
plt.figure(figsize=(10, 6))
sns.barplot(data=reward_df, x='Agent', y='Final Reward')
plt.title('Final rewards for each agent')
plt.xlabel('Agent')
plt.ylabel('Final Reward')
plt.show()
