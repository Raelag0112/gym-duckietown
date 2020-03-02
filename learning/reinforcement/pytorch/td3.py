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
    def __init__(self, state_dim, action_dim, max_action, with_per = False, alpha = 0.7, epsilon = 1e-8):
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
        
        self.with_per = with_per
        self.alpha = alpha
        self.epsilon = epsilon

    def predict(self, state):

        state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, replay_buffer, discount=0.99, tau=0.001):

        # Sample replay buffer
        if not self.with_per:
            experiences = replay_buffer.sample()
            
        else:
            experiences , priorities_list, indices = replay_buffer.sample()
            priorities = torch.from_numpy(np.array(priorities_list)).float().to(device)
            # Calculate importance-sampling weights
            probs = priorities / replay_buffer.priority_sum()
            weights = (replay_buffer.batch_size * probs)**(-replay_buffer.beta)
            weights /= torch.max(weights)

        state = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        action = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        reward = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        next_state = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        done = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        # Compute the target Q value
        target_1_Q = self.critic_target_1(next_state, self.actor_target(next_state))

        target_2_Q = self.critic_target_2(next_state, self.actor_target(next_state))

        target_Q = reward + ((1-done) * discount * torch.min(target_1_Q.view(-1),target_2_Q.view(-1))).detach()

        # Get current Q estimate
        current_Q_1 = self.critic_1(state, action).view(-1)

        current_Q_2 = self.critic_2(state, action).view(-1)
        
        if self.with_per:
            # Update priorities
            td_error_1 = abs(target_Q - current_Q_1)
            td_error_2 = abs(target_Q - current_Q_2)
            updated_priorities = torch.min(td_error_1,td_error_2) + self.epsilon
            replay_buffer.set_priorities(indices, updated_priorities**self.alpha)
            replay_buffer.current_priority = max(replay_buffer.current_priority, torch.max(updated_priorities))
            
            # Compute critic loss
            critic_1_loss = torch.mean(weights * td_error_1**2)

            critic_2_loss = torch.mean(weights * td_error_2**2)
        else:
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
