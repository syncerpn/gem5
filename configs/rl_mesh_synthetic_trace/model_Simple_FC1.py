import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Input -> Output
class Simple_FC1(nn.Module):

    def __init__(self, state_dim, action_d_dim, action_p_dim):

        super(Simple_FC1, self).__init__()
        self.state_dim = state_dim

        self.imm_dims = [256, 128, 64]
        self.imm_dim_last = self.imm_dims[-1]

        self.imms = []
        imm_input_dim = self.state_dim

        for imm_dim in self.imm_dims:
            self.imms.append(nn.Linear(imm_input_dim, imm_dim))
            self.imms[-1].bias.data.fill_(0.01)
            nn.init.xavier_uniform_(self.imms[-1].weight)
            imm_input_dim = imm_dim

        self.action_d_dim = action_d_dim
        self.action_p_dim = action_p_dim

        # Output
        self.action_d_head = nn.Linear(self.imm_dim_last, action_d_dim)
        self.action_d_head.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.action_d_head.weight)

        self.action_p_head = nn.Linear(self.imm_dim_last, action_p_dim)
        self.action_p_head.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.action_p_head.weight)

        self.value_head = nn.Linear(self.imm_dim_last, 1)
        self.value_head.bias.data.fill_(0.01)
        nn.init.xavier_uniform_(self.value_head.weight)

    def forward(self, x):
        """
        forward of both actor and critic
        """
        for imm_layer in self.imms:
            x = F.sigmoid(imm_layer(x))
        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_d_prob = F.softmax(self.action_d_head(x), dim=-1)
        action_p_prob = F.softmax(self.action_p_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_value = self.value_head(x)

        return action_d_prob, action_p_prob, state_value

    def save(self, save_dir, i_episode):
        torch.save(self.state_dict(), save_dir + 'Simple_FC1_' + str(i_episode) + '.pt')
        print('[INFO] Models saved successfully')

    def load(self, model_file):
        ckpt = torch.load(model_file)
        self.load_state_dict(ckpt)