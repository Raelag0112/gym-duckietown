import random

import gym
import numpy as np
import torch

from collections import deque, namedtuple

from reinforcement.pytorch.SumTree import SumTree

#from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree

def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def evaluate_policy(env, policy, eval_episodes=10, max_timesteps=500):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        step = 0
        while not done and step < max_timesteps:
            action = policy.predict(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            step += 1

    avg_reward /= eval_episodes

    return avg_reward


# Codes based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
          action_size (int): dimension of each action
          buffer_size (int): maximum size of buffer
          batch_size (int): size of each training batch
          seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        return experiences

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


### Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    """Fixed-size buffer for storing experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, initial_beta=0.5, delta_beta=1e-5):
        """Initialize a ReplayBuffer object.

        Params
        ======
          buffer_size (int): maximum size of buffer
          batch_size (int): size of each training batch
          seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
        self.priorities = SumTree(capacity=buffer_size)
        
        self.current_priority = 1.0
        
        self.initial_beta = initial_beta
        self.delta_beta = delta_beta
        self.beta = initial_beta

    def add(self, state, action, reward, next_state, done):
        """Add an experience to memory.
        Args:
            state (Tensor): Current state
            action (int): Chosen action
            reward (float): Resulting reward
            next_state (Tensor): State after action
            done (bool): True if terminal state
            priority (float): Priority of experience (abs TD-error)
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
        self.priorities.append(self.current_priority)
        
        self.beta += self.delta_beta

    def sample(self):
            
        indices = self.priorities.sample(self.batch_size)

        # TODO: Make this whole indexing and unpacking thing more efficient
        experiences = [self.memory[i] for i in indices]

        priorities_list = [self.priorities.get(i) for i in indices]

        return experiences , priorities_list, indices

    def priority_sum(self):
        return self.priorities.total_sum()

    def set_priorities(self, i, p):
        # NB. Works with multiple indices and priorities

        self.priorities.set_multiple(i, p)

    def __len__(self):
        """Returns the current number of stored experiences.
        Returns:
            int: Number of stored experiences"""
        return len(self.memory)
