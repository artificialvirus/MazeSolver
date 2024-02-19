import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pygame
import pickle
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
import torch.nn.functional as F

# done
        
#Use a more sophisticated model
# CNN model (considering the maze as an image)
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lr=0.001, gamma=0.85, epsilon=0.5, buffer_size=10000, batch_size=64, update_target_every=5):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.learn_step_counter = 0  # For updating the target network

        self.memory = ReplayBuffer(buffer_size)
        self.model = QNetwork(state_size, action_size, hidden_size)
        self.target_model = QNetwork(state_size, action_size, hidden_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.loss_history = []  # Stores loss for each update
        self.avg_q_value_history = []  # Stores average Q-value for each episode

    def choose_action(self, state, test=False):
        if test or np.random.rand() > self.epsilon:  # No exploration if testing
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
            return q_values.argmax().item()
        else:
            return random.randrange(self.action_size)


    def update(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        curr_q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_model(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Track average Q-values
        avg_q_value = curr_q_values.mean().item()
        self.avg_q_value_history.append(avg_q_value)

        loss = self.criterion(curr_q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store loss
        self.loss_history.append(loss.item())

        if self.learn_step_counter % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        self.learn_step_counter += 1

    def report_performance_metrics(self):
        # Calculate and return average loss and Q-value for the current episode
        avg_loss = sum(self.loss_history) / len(self.loss_history) if self.loss_history else 0
        avg_q_value = sum(self.avg_q_value_history) / len(self.avg_q_value_history) if self.avg_q_value_history else 0
        self.loss_history.clear()  # Reset for the next episode
        self.avg_q_value_history.clear()  # Reset for the next episode
        return avg_loss, avg_q_value

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

class SarsaAgent:
    def __init__(self, num_states, num_actions, alpha=0.02, gamma=0.80, epsilon=0.5):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # Use a dictionary to handle dynamic state-action space

    def choose_action(self, state, test=False):
        state_key = (state[0], state[1])
        # Ensure there is an entry for the current state in Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)

        if test:
            # Exploitation only for testing, no random choice.
            action = np.argmax(self.q_table[state_key])
        else:
            # Epsilon-greedy action selection for training
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.num_actions)  # Exploration
            else:
                action = np.argmax(self.q_table[state_key])  # Exploitation
        return action
    

    def update(self, state, action, reward, next_state, next_action):
        state_key = (state[0], state[1])
        next_state_key = (next_state[0], next_state[1])
        # Initialize state-action space if not exist
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.num_actions)
        # Sarsa update
        q_old = self.q_table[state_key][action]
        next_q = self.q_table[next_state_key][next_action]
        self.q_table[state_key][action] = q_old + self.alpha * (reward + self.gamma * next_q - q_old)

class EnhancedEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.screen = None
        self.state = (1, 1)  # Starting position, you can adjust this
        self.goal = (23, 23)  # Goal position, you can adjust this
        self.done = False
        self.maze = np.zeros((self.height, self.width), dtype=np.int8)
        self.visit_counts = np.zeros((height, width), dtype=np.int32)  # Track visits for each cell
        self.init_pygame()
        self.generate_environment()

    def init_pygame(self):
        pygame.init()
        self.cell_size = 20
        self.screen_size = (self.width * self.cell_size, self.height * self.cell_size)
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Enhanced RL Environment")

    def generate_environment(self):
        # Fill the maze with walls (1s)
        self.maze.fill(1)

        # Manually create paths in the maze (set cells to 0)
        # Example: creating a simple path from top-left to bottom-right
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                if x % 2 == 0 or y % 2 == 0:  # Adjust conditions for your design
                    self.maze[y, x] = 0

        # Optionally, create specific rooms or larger areas
        self.maze[2:4, 2:4] = 0  # Example room
        self.maze[18:22, 18:22] = 0  # Another example room

        # Set the goal position
        self.maze[self.goal] = 2

    def get_surrounding_state(self, x, y):
        """Returns the status of adjacent cells: [up, right, down, left]"""
        surroundings = []
        for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:  # Up, Right, Down, Left
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                surroundings.append(self.maze[ny][nx] == 1)  # 1 if wall, 0 if path
            else:
                surroundings.append(1)  # Treat out-of-bounds as walls
        return surroundings

    def reset(self):
        self.done = False
        self.state = (1, 1)  # Reset to starting position
        surrounding_state = self.get_surrounding_state(*self.state)
        return self.state + tuple(surrounding_state)

    def step(self, action):
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        next_x, next_y = self.state[0] + dx, self.state[1] + dy
        self.visit_counts[self.state[1], self.state[0]] += 1  # Increment visit count for the current state

        # Initialize reward variable before using it
        reward = 0

        # Adjust the reward based on the environment's response to the action
        if 0 <= next_x < self.width and 0 <= next_y < self.height:
            if self.maze[next_y][next_x] != 1:
                self.state = (next_x, next_y)
                if self.maze[next_y][next_x] == 2:
                    reward = 100  # Reward for reaching the goal
                    self.done = True
                else:
                    reward = -1  # Slight penalty for each move to encourage efficiency
            else:
                reward = -5  # Lesser penalty for hitting a wall to encourage exploration
        else:
            reward = -10  # Higher penalty for attempting to move outside the maze bounds

        intrinsic_reward = self.calculate_intrinsic_reward(self.state)
        reward += intrinsic_reward  # Add intrinsic reward to the existing reward

        surrounding_state = self.get_surrounding_state(next_x, next_y)
        return (next_x, next_y) + tuple(surrounding_state), reward, self.done, {}

    
    def calculate_intrinsic_reward(self, state):
        visit_count = self.visit_counts[state[1], state[0]]
        # Simple novelty-based intrinsic reward - inversely proportional to visit count
        if visit_count == 0:
            return 10  # Higher reward for completely new states
        else:
            return 1 / visit_count  # Decreases as state becomes less novel

    def render(self):
        self.screen.fill((255, 255, 255))
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                color = (0, 0, 0) if self.maze[y, x] == 1 else (255, 255, 255)
                if self.state == (x, y):
                    color = (0, 255, 0)
                elif self.maze[y, x] == 2:
                    color = (255, 0, 0)
                pygame.draw.rect(self.screen, color, rect)
        pygame.display.flip()

# Done

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

# Not imported 

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


# Execute Bayesian optimization to find the best hyperparameters
result_qlearning = gp_minimize(objective_qlearning, space, n_calls=20, random_state=0)
print("Best parameters for QLearningAgent: {}".format(result_qlearning.x))
print("Best average reward: {}".format(-result_qlearning.fun))

result_sarsa = gp_minimize(objective_sarsa, space, n_calls=20, random_state=0)
print("Best parameters for SarsaAgent: {}".format(result_sarsa.x))
print("Best average reward: {}".format(-result_sarsa.fun))

result_dqn = gp_minimize(objective_dqn, space_dqn, n_calls=10, random_state=0)
print("Best parameters: {}".format(result_dqn.x))
print("Best average reward: {}".format(-result_dqn.fun))


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


# To save the current state of environments
save_environment(env, 'current_environment.pkl')
save_environment(testing_env, 'current_testing_environment.pkl')

# To load the environment (use when needed)
env = load_environment('current_environment.pkl')
testing_env = load_environment('current_testing_environment.pkl')

# Training and testing
episode_rewards = {agent_name: [] for agent_name in agents}
episode_steps = {agent_name: [] for agent_name in agents}
learning_speeds = {agent_name: 0 for agent_name in agents}
rewards = {agent_name: [] for agent_name in agents}

satisfactory_performance = 25  # threshold for satisfactory performance
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


