#!/usr/bin/env python3
import random
import numpy as np

class ReplayBuffer():

    def __init__(self, capacity) -> None:
        self.memory = []
        self.position = 0
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        # self.memory[self.position] = transition
        # self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), \
            np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.memory)
