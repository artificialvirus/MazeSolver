# This file contains the EnhancedEnvironment class, which is an extension of the Environment class.
# File: environment/mazes/enhanced_environment.py

import numpy as np
import pygame

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
