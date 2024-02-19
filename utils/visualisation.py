# This file contains the code for the visualisation of the environment and the agent's behavior
# File: utils/visualisation.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_environment(environment, agent=None, max_steps=1000):
    # Plot the environment and the agent's behavior
    fig, ax = plt.subplots()
    ax.imshow(environment.maze, cmap='binary')
    ax.set_xticks([])
    ax.set_yticks([])
    state = environment.reset()
    for _ in range(max_steps):
        action = agent.act(state) if agent is not None else environment.sample_action()
        state, _, done, _ = environment.step(action)
        ax.imshow(environment.maze, cmap='binary')
        ax.plot(state[1], state[0], 'ro', markersize=10)
        fig.canvas.draw()
        plt.pause(0.1)
        if done:
            break
    plt.close(fig)

def plot_rewards(episode_rewards):
    # Plot the rewards obtained by the agents during training
    plt.figure()
    for agent_name, rewards in episode_rewards.items():
        plt.plot(rewards, label=agent_name)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Episode Rewards')
    plt.legend()
    plt.show()

def plot_test_results(rewards):
    # Plot the average rewards obtained by the agents during testing
    plt.figure()
    plt.bar(rewards.keys(), rewards.values())
    plt.xlabel('Agent')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards during Testing')
    plt.show()

def plot_learning_speed(learning_speeds):
    # Plot the learning speed of the agents
    plt.figure()
    plt.bar(learning_speeds.keys(), learning_speeds.values())
    plt.xlabel('Agent')
    plt.ylabel('Learning Speed')
    plt.title('Learning Speed of the Agents')
    plt.show()

def plot_performance_metrics(reward_df, steps_df, learning_speed_df):
    # Plot the performance metrics of the agents
    fig, axs = plt.subplots(3, figsize=(10, 15))
    sns.lineplot(data=reward_df, x=reward_df.index, y='Total Reward', hue='Agent', ax=axs[0])
    axs[0].set_title('Total reward per episode for each agent')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')
    sns.lineplot(data=steps_df, x=steps_df.index, y='Total Steps', hue='Agent', ax=axs[1])
    axs[1].set_title('Total steps per episode for each agent')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Total Steps')
    sns.barplot(data=learning_speed_df, x='Agent', y='Learning Speed', ax=axs[2])
    axs[2].set_title('Learning speed for each agent')
    axs[2].set_xlabel('Agent')
    axs[2].set_ylabel('Learning Speed')
    plt.tight_layout()
    plt.show()

def plot_final_rewards(reward_df):
    # Plot the final rewards obtained by the agents
    plt.figure()
    sns.barplot(data=reward_df, x='Agent', y='Final Reward')
    plt.title('Final rewards for each agent')
    plt.xlabel('Agent')
    plt.ylabel('Final Reward')
    plt.show()

def plot_generalization_test_results(generalization_scores):
    # Plot the generalization test results
    plt.figure()
    plt.bar(generalization_scores.keys(), generalization_scores.values())
    plt.xlabel('Agent')
    plt.ylabel('Average Reward')
    plt.title('Generalization Test Average Rewards')
    plt.show()

def plot_hyperparameter_optimization_results(result_qlearning, result_sarsa, result_dqn):
    # Plot the results of the hyperparameter optimization
    plt.figure()
    plt.plot(result_qlearning.func_vals, label='QLearning')
    plt.plot(result_sarsa.func_vals, label='Sarsa')
    plt.plot(result_dqn.func_vals, label='DQN')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.title('Hyperparameter Optimization Results')
    plt.legend()
    plt.show()

def plot_best_hyperparameters(result_qlearning, result_sarsa, result_dqn):
    # Plot the best hyperparameters
    plt.figure()
    plt.bar(['QLearning', 'Sarsa', 'DQN'], [-result_qlearning.fun, -result_sarsa.fun, -result_dqn.fun])
    plt.xlabel('Agent')
    plt.ylabel('Average Reward')
    plt.title('Best Hyperparameters for Each Agent')
    plt.show()  

def plot_steps(steps_df):
    # Plot the steps taken by the agents
    plt.figure()
    sns.lineplot(data=steps_df, x=steps_df.index, y='Total Steps', hue='Agent')
    plt.title('Total steps per episode for each agent')
    plt.xlabel('Episode')
    plt.ylabel('Total Steps')
    plt.show()
