#!/usr/bin/env python3
"""
model_as_modal/model_as_modal/policy.py

PPO Policy Network for Model as Modal
Designed to handle combined physical and perception state
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class PPOPolicy(nn.Module):
    """
    PPO Policy Network for Model as Modal
    Designed to handle combined physical and perception state
    """
    def __init__(self, state_dim=13, action_dim=4, hidden_dim=128):
        super(PPOPolicy, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Separate processing for perception features (Model as Modal)
        self.perception_net = nn.Sequential(
            nn.Linear(4, 32),  # 4 detection features
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # Combined processing
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head (mean of action distribution)
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Log standard deviation (learnable)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """Forward pass through the network"""
        # Handle both single states and batches
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Split state into physical and perception components
        physical_state = state[:, :9]  # position, velocity, angular_velocity
        perception_state = state[:, 9:13]  # detection statistics
        
        # Process physical state through main network
        features = self.feature_net(state)
        
        # Process perception state separately (Model as Modal innovation)
        perception_features = self.perception_net(perception_state)
        
        # Combine features
        combined = torch.cat([features, perception_features], dim=1)
        final_features = self.combined_net(combined)
        
        # Get policy and value outputs
        action_mean = torch.tanh(self.policy_mean(final_features))  # Actions in [-1, 1]
        action_std = torch.exp(torch.clamp(self.log_std, -5, 2))
        value = self.value_head(final_features)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """Get action from the policy"""
        state_tensor = torch.FloatTensor(state)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        with torch.no_grad():
            action_mean, action_std, value = self.forward(state_tensor)
            
            if deterministic:
                action = action_mean
            else:
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                
            # Ensure actions are in valid range
            action = torch.clamp(action, -1, 1)
            
            # Calculate log probability for PPO
            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.numpy()[0], log_prob.numpy()[0], value.numpy()[0]
    
    def evaluate_actions(self, states, actions):
        """Evaluate actions for PPO update"""
        action_mean, action_std, values = self.forward(states)
        
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        
        return values.squeeze(), log_probs, dist_entropy