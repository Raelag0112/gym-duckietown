import functools
import operator

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from numpy import dtype

#Code based on:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
        
        self.optimizer = torch.optim.RMSprop(self.parameters())

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    
    def optimize_model(self, memory, target_net, batch_size = 64, gamma = 0.999):
        if len(memory.storage) < batch_size:
            return
        # Sample replay buffer
        # sample is a dictionary of stacked states, actions...
        sample = memory.sample(batch_size, flat=False)
        state = torch.FloatTensor(sample["state"]).to(device)
        action = torch.FloatTensor(sample["action"]).long().to(device)
        next_state = torch.FloatTensor(sample["next_state"]).to(device)
        done = torch.FloatTensor(1 - sample["done"]).to(device)
        reward = torch.FloatTensor(sample["reward"]).to(device)
    
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              sample["next_state"])), device=device, dtype=torch.bool)
        non_final_next_states = torch.FloatTensor([]).to(device)
        templist = []
        for s in sample["next_state"]:
            s = torch.FloatTensor(s).to(device)
            if s is not None:
                templist.append(s)
        non_final_next_states = torch.stack(templist).to(device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        action = torch.LongTensor(action);
        state_action_values = self(state).gather(1, action)
    
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward
    
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def predict(self, state):
        # just making sure the state has the correct format, otherwise the prediction doesn't work
        assert state.shape[0] == 3

        state = np.expand_dims(state, axis = 0)
        state = torch.FloatTensor(state).to(device)
        return self(state).max(1)[1].view(1, 1).cpu().data.numpy().flatten()
    
    

