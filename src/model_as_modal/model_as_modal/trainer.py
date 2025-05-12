#!/usr/bin/env python3
"""
model_as_modal/model_as_modal/trainer.py

PPO Training implementation for Model as Modal using Gymnasium
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import time


class PPOTrainer:
    """
    PPO Training implementation for Model as Modal using Gymnasium
    """
    def __init__(self, env, policy, lr=3e-4, gamma=0.99, gae_lambda=0.95, 
                 eps_clip=0.2, value_clip=0.2, ppo_epochs=10, batch_size=64):
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.value_clip = value_clip
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_detection_confidences = []
        
        # Rolling buffers for recent statistics
        self.recent_rewards = deque(maxlen=100)
        self.recent_confidences = deque(maxlen=100)
        
    def collect_rollouts(self, n_steps=2048):
        """Collect experience for PPO training"""
        obs_buf = []
        act_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []
        logp_buf = []
        
        obs, _ = self.env.reset()
        done = False
        episode_reward = 0
        episode_detection_confidences = []
        episode_start = time.time()
        
        for step in range(n_steps):
            # Get action from policy
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(obs)
            
            # Take action in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)
            done_buf.append(done)
            val_buf.append(value)
            logp_buf.append(log_prob)
            
            # Track episode statistics
            episode_reward += reward
            episode_detection_confidences.append(info.get('detection_confidence', 0))
            
            obs = next_obs
            
            if done:
                # Episode finished
                episode_length = len(episode_detection_confidences)
                episode_time = time.time() - episode_start
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_detection_confidences.append(np.mean(episode_detection_confidences))
                
                self.recent_rewards.append(episode_reward)
                self.recent_confidences.append(np.mean(episode_detection_confidences))
                
                # Reset for next episode
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_detection_confidences = []
                episode_start = time.time()
        
        # Convert to numpy arrays
        obs_buf = np.array(obs_buf)
        act_buf = np.array(act_buf)
        rew_buf = np.array(rew_buf)
        done_buf = np.array(done_buf)
        val_buf = np.array(val_buf)
        logp_buf = np.array(logp_buf)
        
        # Calculate advantages and returns
        adv_buf, ret_buf = self._calculate_gae(rew_buf, val_buf, done_buf)
        
        return {
            'obs': obs_buf,
            'act': act_buf,
            'ret': ret_buf,
            'adv': adv_buf,
            'logp': logp_buf,
            'val': val_buf
        }
    
    def _calculate_gae(self, rewards, values, dones):
        """Calculate Generalized Advantage Estimation"""
        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        returns = np.zeros(n_steps)
        
        # Get final value
        final_obs = self.env._get_observation()
        final_obs_tensor = torch.FloatTensor(final_obs).unsqueeze(0)
        with torch.no_grad():
            _, _, final_value = self.policy(final_obs_tensor)
            final_value = final_value.item()
        
        # Calculate GAE
        last_gae = 0
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = final_value
                next_done = dones[t]
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * last_gae
            
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(self, rollout_data):
        """Update policy using PPO"""
        # Convert to tensors
        obs_tensor = torch.FloatTensor(rollout_data['obs'])
        act_tensor = torch.FloatTensor(rollout_data['act'])
        ret_tensor = torch.FloatTensor(rollout_data['ret'])
        adv_tensor = torch.FloatTensor(rollout_data['adv'])
        old_logp_tensor = torch.FloatTensor(rollout_data['logp'])
        old_val_tensor = torch.FloatTensor(rollout_data['val'])
        
        # Normalize advantages
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        
        # Create dataset
        dataset_size = len(obs_tensor)
        indices = np.arange(dataset_size)
        
        # PPO update
        for epoch in range(self.ppo_epochs):
            # Shuffle indices
            np.random.shuffle(indices)
            
            # Mini-batch updates
            for start_idx in range(0, dataset_size, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                batch_obs = obs_tensor[batch_indices]
                batch_act = act_tensor[batch_indices]
                batch_ret = ret_tensor[batch_indices]
                batch_adv = adv_tensor[batch_indices]
                batch_old_logp = old_logp_tensor[batch_indices]
                batch_old_val = old_val_tensor[batch_indices]
                
                # Forward pass
                action_mean, action_std, values = self.policy(batch_obs)
                dist = Normal(action_mean, action_std)
                log_probs = dist.log_prob(batch_act).sum(dim=-1)
                
                # Calculate ratio
                ratio = torch.exp(log_probs - batch_old_logp)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (with clipping)
                values_clipped = batch_old_val + torch.clamp(
                    values.squeeze() - batch_old_val,
                    -self.value_clip,
                    self.value_clip
                )
                value_loss_1 = (values.squeeze() - batch_ret).pow(2)
                value_loss_2 = (values_clipped - batch_ret).pow(2)
                value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                
                # Entropy bonus
                entropy_loss = -dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
    
    def train(self, total_timesteps=1000000, n_steps=2048):
        """Main training loop"""
        n_updates = total_timesteps // n_steps
        
        for update in range(1, n_updates + 1):
            update_start_time = time.time()
            
            # Collect rollouts
            rollout_data = self.collect_rollouts(n_steps)
            
            # Update policy
            self.update(rollout_data)
            
            # Log progress
            if update % 5 == 0:
                avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0
                avg_detection = np.mean(self.recent_confidences) if self.recent_confidences else 0
                
                print(f"Update {update}/{n_updates}")
                print(f"  Average Reward: {avg_reward:.2f}")
                print(f"  Average Detection Confidence: {avg_detection:.3f}")
                print(f"  Episodes completed: {len(self.episode_rewards)}")
                print(f"  Update time: {time.time() - update_start_time:.2f}s")
                print("---")
                
                # Save checkpoint
                if update % 50 == 0:
                    torch.save(self.policy.state_dict(), f'model_as_modal_checkpoint_{update}.pth')
                    print(f"Checkpoint saved: model_as_modal_checkpoint_{update}.pth")
        
        # Save final model
        torch.save(self.policy.state_dict(), 'model_as_modal_final.pth')
        return self.episode_rewards, self.episode_detection_confidences
    
    def train_episode(self):
        """Train for one episode (used by training script)"""
        # Collect experience for one episode
        obs, _ = self.env.reset()
        done = False
        episode_reward = 0
        episode_detection_confidences = []
        
        states, actions, rewards, values, log_probs = [], [], [], [], []
        
        while not done:
            # Get action
            action, log_prob, value = self.policy.get_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            
            episode_reward += reward
            episode_detection_confidences.append(info.get('detection_confidence', 0))
            
            obs = next_obs
        
        # Calculate advantages and update policy
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        values = np.array(values)
        log_probs = np.array(log_probs)
        
        # Simple advantage calculation for single episode
        advantages = rewards - values
        returns = rewards
        
        # Update policy
        rollout_data = {
            'obs': states,
            'act': actions,
            'ret': returns,
            'adv': advantages,
            'logp': log_probs,
            'val': values
        }
        self.update(rollout_data)
        
        # Return episode statistics
        episode_stats = {
            'mean_detection_confidence': np.mean(episode_detection_confidences),
            'mean_stability': episode_reward  # Simplified metric
        }
        
        return episode_reward, episode_stats