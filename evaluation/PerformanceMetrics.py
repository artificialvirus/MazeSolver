# This file contains the code for the performance metrics used to evaluate the agents
# File: evaluation/PerformanceMetrics.py

from environment.mazes.enhanced_environment import EnhancedEnvironment


class PerformanceMetrics:
    def __init__(self, environment, agent, max_steps=1000):
        self.environment = environment
        self.agent = agent
        self.max_steps = max_steps

    def run(self, n=100):
        # Run the agent in the environment n times and return the average reward
        rewards = []
        for _ in range(n):
            state = self.environment.reset()
            total_reward = 0
            for _ in range(self.max_steps):
                action = self.agent.act(state)
                state, reward, done, _ = self.environment.step(action)
                total_reward += reward
                if done:
                    break
            rewards.append(total_reward)
        return sum(rewards) / n
    
    def run_episode(self):
        # Run the agent in the environment for one episode and return the total reward
        state = self.environment.reset()
        total_reward = 0
        for _ in range(self.max_steps):
            action = self.agent.act(state)
            state, reward, done, _ = self.environment.step(action)
            total_reward += reward
            if done:
                break
        return total_reward
    
    def run_episode_steps(self):
        # Run the agent in the environment for one episode and return the number of steps
        state = self.environment.reset()
        num_steps = 0
        for _ in range(self.max_steps):
            action = self.agent.act(state)
            state, reward, done, _ = self.environment.step(action)
            num_steps += 1
            if done:
                break
        return num_steps
    
    def run_episode_steps_and_reward(self):
        # Run the agent in the environment for one episode and return the number of steps and the total reward
        state = self.environment.reset()
        total_reward = 0
        num_steps = 0
        for _ in range(self.max_steps):
            action = self.agent.act(state)
            state, reward, done, _ = self.environment.step(action)
            total_reward += reward
            num_steps += 1
            if done:
                break
        return num_steps, total_reward
    
    def run_episode_steps_and_reward_and_done(self):
        # Run the agent in the environment for one episode and return the number of steps, the total reward, and whether the episode is done
        state = self.environment.reset()
        total_reward = 0
        num_steps = 0
        done = False
        for _ in range(self.max_steps):
            action = self.agent.act(state)
            state, reward, done, _ = self.environment.step(action)
            total_reward += reward
            num_steps += 1
            if done:
                break
        return num_steps, total_reward, done
    
    def run_episode_steps_and_reward_and_done_and_info(self):
        # Run the agent in the environment for one episode and return the number of steps, the total reward, whether the episode is done, and the info dictionary
        state = self.environment.reset()
        total_reward = 0
        num_steps = 0
        done = False
        info = {}
        for _ in range(self.max_steps):
            action = self.agent.act(state)
            state, reward, done, info = self.environment.step(action)
            total_reward += reward
            num_steps += 1
            if done:
                break
        return num_steps, total_reward, done, info
    
    def run_episode_steps_and_reward_and_done_and_info_and_state(self):
        # Run the agent in the environment for one episode and return the number of steps, the total reward, whether the episode is done, the info dictionary, and the final state
        state = self.environment.reset()
        total_reward = 0
        num_steps = 0
        done = False
        info = {}
        for _ in range(self.max_steps):
            action = self.agent.act(state)
            state, reward, done, info = self.environment.step(action)
            total_reward += reward
            num_steps += 1
            if done:
                break
        return num_steps, total_reward, done, info, state
    
    def run_episode_steps_and_reward_and_done_and_info_and_state_and_action(self):
        # Run the agent in the environment for one episode and return the number of steps, the total reward, whether the episode is done, the info dictionary, the final state, and the final action
        state = self.environment.reset()
        total_reward = 0
        num_steps = 0
        done = False
        info = {}
        action = None
        for _ in range(self.max_steps):
            action = self.agent.act(state)
            state, reward, done, info = self.environment.step(action)
            total_reward += reward
            num_steps += 1
            if done:
                break
        return num_steps, total_reward, done, info, state, action
    
    def run_episode_steps_and_reward_and_done_and_info_and_state_and_action_and_next_state(self):
        # Run the agent in the environment for one episode and return the number of steps, the total reward, whether the episode is done, the info dictionary, the final state, the final action, and the next state
        state = self.environment.reset()
        total_reward = 0
        num_steps = 0
        done = False
        info = {}
        action = None
        next_state = None
        for _ in range(self.max_steps):
            action = self.agent.act(state)
            next_state, reward, done, info = self.environment.step(action)
            total_reward += reward
            num_steps += 1
            if done:
                break
        return num_steps, total_reward, done, info, state, action, next_state
    
    def run_episode_steps_and_reward_and_done_and_info_and_state_and_action_and_next_state_and_reward(self):
        # Run the agent in the environment for one episode and return the number of steps, the total reward, whether the episode is done, the info dictionary, the final state, the final action, the next state, and the reward
        state = self.environment.reset()
        total_reward = 0
        num_steps = 0
        done = False
        info = {}
        action = None
        next_state = None
        reward = 0
        for _ in range(self.max_steps):
            action = self.agent.act(state)
            next_state, reward, done, info = self.environment.step(action)
            total_reward += reward
            num_steps += 1
            if done:
                break
        return num_steps, total_reward, done, info, state, action, next_state, reward
    
    def run_episode_steps_and_reward_and_done_and_info_and_state_and_action_and_next_state_and_reward_and_done(self):
        # Run the agent in the environment for one episode and return the number of steps, the total reward, whether the episode is done, the info dictionary, the final state, the final action, the next state, the reward, and whether the episode is done
        state = self.environment.reset()
        total_reward = 0
        num_steps = 0
        done = False
        info = {}
        action = None
        next_state = None
        reward = 0
        for _ in range(self.max_steps):
            action = self.agent.act(state)
            next_state, reward, done, info = self.environment.step(action)
            total_reward += reward
            num_steps += 1
            if done:
                break
        return num_steps, total_reward, done, info, state, action, next_state, reward, done
    
    def run_episode_steps_and_reward_and_done_and_info_and_state_and_action_and_next_state_and_reward_and_done_and_info(self):
        # Run the agent in the environment for one episode and return the number of steps, the total reward, whether the episode is done, the info dictionary, the final state, the final action, the next state, the reward, whether the episode is done, and the info dictionary
        state = self.environment.reset()
        total_reward = 0
        num_steps = 0
        done = False
        info = {}
        action = None
        next_state = None
        reward = 0
        for _ in range(self.max_steps):
            action = self.agent.act(state)
            next_state, reward, done, info = self.environment.step(action)
            total_reward += reward
            num_steps += 1
            if done:
                break
        return num_steps, total_reward, done, info, state, action, next_state, reward, done, info
    
    def run_episode_steps_and_reward_and_done_and_info_and_state_and_action_and_next_state_and_reward_and_done_and_info_and_env(self):
        # Run the agent in the environment for one episode and return the number of steps, the total reward, whether the episode is done, the info dictionary, the final state, the final action, the next state, the reward, whether the episode is done, the info dictionary, and the environment
        state = self.environment.reset()
        total_reward = 0
        num_steps = 0
        done = False
        info = {}
        action = None
        next_state = None
        reward = 0
        for _ in range(self.max_steps):
            action = self.agent.act(state)
            next_state, reward, done, info = self.environment.step(action)
            total_reward += reward
            num_steps += 1
            if done:
                break
        return num_steps, total_reward, done, info, state, action, next_state, reward, done, info, self.environment
    
    def run_episode_steps_and_reward_and_done_and_info_and_state_and_action_and_next_state_and_reward_and_done_and_info_and_env_and_agent(self):

        # Run the agent in the environment for one episode and return the number of steps, the total reward, whether the episode is done, the info dictionary, the final state, the final action, the next state, the reward, whether the episode is done, the info dictionary, the environment, and the agent
        state = self.environment.reset()
        total_reward = 0
        num_steps = 0
        done = False
        info = {}
        action = None
        next_state = None
        reward = 0
        for _ in range(self.max_steps):
            action = self.agent.act(state)
            next_state, reward, done, info = self.environment.step(action)
            total_reward += reward
            num_steps += 1
            if done:
                break
        return num_steps, total_reward, done, info, state, action, next_state, reward, done, info, self.environment, self.agent
    
    def run_episode_steps_and_reward_and_done_and_info_and_state_and_action_and_next_state_and_reward_and_done_and_info_and_env_and_agent_and_max_steps(self):

        # Run the agent in the environment for one episode and return the number of steps, the total reward, whether the episode is done, the info dictionary, the final state, the final action, the next state, the reward, whether the episode is done, the info dictionary, the environment, the agent, and the maximum number of steps
        state = self.environment.reset()
        total_reward = 0
        num_steps = 0
        done = False
        info = {}
        action = None
        next_state = None
        reward = 0
        for _ in range(self.max_steps):
            action = self.agent.act(state)
            next_state, reward, done, info = self.environment.step(action)
            total_reward += reward
            num_steps += 1
            if done:
                break
        return num_steps, total_reward, done, info, state, action, next_state, reward, done, info, self.environment, self.agent, self.max_steps
