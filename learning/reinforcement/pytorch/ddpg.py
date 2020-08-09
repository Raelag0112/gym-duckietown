import functools
import operator

import numpy as np
import torch
import torch.nn.functional as F

from reinforcement.pytorch.actor import ActorCNN
from reinforcement.pytorch.critic import CriticCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class DDPG(object):

    def __init__(self, state_dim, action_dim, max_action, with_per = False, alpha = 0.7, epsilon = 1e-8, grad_clipping = False):
        super(DDPG, self).__init__()

        print("Starting DDPG init")
        self.state_dim = state_dim

        self.actor = ActorCNN(action_dim, max_action).to(device)
        self.actor_target = ActorCNN(action_dim, max_action).to(device)

        print("Initialized Actor")
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        print("Initialized Target+Opt [Actor]")

        self.critic = CriticCNN(action_dim).to(device)
        self.critic_target = CriticCNN(action_dim).to(device)

        print("Initialized Critic")
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        print("Initialized Target+Opt [Critic]")
        
        self.with_per = with_per
        self.alpha = alpha
        self.epsilon = epsilon

        self.grad_clipping = grad_clipping
        
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
        target_Q = self.critic_target(next_state, self.actor_target(next_state))

        target_Q = reward + ((1 - done) * discount * target_Q.view(-1)).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action).view(-1)
        
        if self.with_per:
            # Update priorities
            td_error = target_Q - current_Q
            updated_priorities = abs(td_error) + self.epsilon
            replay_buffer.set_priorities(indices, updated_priorities**self.alpha)
            replay_buffer.current_priority = max(replay_buffer.current_priority, torch.max(updated_priorities))
            
            # Compute critic loss
            critic_loss = torch.mean(weights * td_error**2)
        else:
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clipping:
                for param in self.critic.nn.parameters():
                    param.grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clipping:
                for param in self.actor.nn.parameters():
                    param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

        self.soft_update(tau)

    def soft_update(self, tau):
        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)



    def save(self, directory, filename):
        save_dict = {
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict()
                }

        torch.save(save_dict, directory + filename)

    def load(self, directory, filename):
        load_dict = torch.load(directory + filename, map_location=device)

        self.actor.load_state_dict(load_dict['actor_state_dict'])
        self.critic.load_state_dict(load_dict['critic_state_dict'])
