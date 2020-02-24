import functools
import operator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reinforcement.pytorch.actor import ActorCNN
from reinforcement.pytorch.critic import CriticCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of  Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        super(TD3, self).__init__()

        self.timestep = 0

        print("Starting TD3 init")
        self.state_dim = state_dim

        self.actor = ActorCNN(action_dim, max_action).to(device)
        self.actor_target = ActorCNN(action_dim, max_action).to(device)

        print("Initialized Actor")

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        print("Initialized Target+Opt [Actor]")

        self.critic_1 = CriticCNN(action_dim).to(device)
        self.critic_target_1 = CriticCNN(action_dim).to(device)
        self.critic_2 = CriticCNN(action_dim).to(device)
        self.critic_target_2 = CriticCNN(action_dim).to(device)

        print("Initialized Critics 1 & 2")

        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters())
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters())

        print("Initialized Target+Opt [Critics]")

    def predict(self, state):

        state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, replay_buffer, discount=0.99, tau=0.001):

        # Sample replay buffer
        experiences = replay_buffer.sample()

        state = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        action = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        reward = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_state = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        done = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        # Compute the target Q value
        target_1_Q = self.critic_target_1(next_state, self.actor_target(next_state))

        target_2_Q = self.critic_target_2(next_state, self.actor_target(next_state))

        target_Q = reward + (done * discount * torch.min(target_1_Q,target_2_Q)).detach()

        # Get current Q estimate
        current_Q_1 = self.critic_1(state, action)

        current_Q_2 = self.critic_2(state, action)

        # Compute critics loss
        critic_1_loss = F.mse_loss(current_Q_1, target_Q)

        critic_2_loss = F.mse_loss(current_Q_2, target_Q)

        # Optimize the critics
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        if self.timestep % 2 == 0:

            # Compute actor loss
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


        self.soft_update(tau)
        self.timestep += 1

    def soft_update(self, tau):
        # Update the frozen target models
        for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, directory, filename):
        save_dict = {
                'actor_state_dict': self.actor.state_dict(),
                'critic_1_state_dict': self.critic_1.state_dict(),
                'critic_2_state_dict': self.critic_2.state_dict()
                }

        torch.save(save_dict, directory + filename)

    def load(self, directory, filename):
        load_dict = torch.load(directory + filename, map_location=device)
        self.actor.load_state_dict(load_dict['actor_state_dict'])
        self.critic_1.load_state_dict(load_dict['critic_1_state_dict'])
        self.critic_2.load_state_dict(load_dict['critic_2_state_dict'])
