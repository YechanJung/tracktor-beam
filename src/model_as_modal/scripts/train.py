#!/usr/bin/env python3
"""
model_as_modal/scripts/train.py

Main training script for Model as Modal
"""

import rclpy
from rclpy.node import Node
import torch
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from model_as_modal.environment import ModelAsModalEnv
from model_as_modal.policy import PPOPolicy
from model_as_modal.trainer import PPOTrainer


class TrainingNode(Node):
    def __init__(self):
        super().__init__('model_as_modal_training')
        
        # Training parameters
        self.declare_parameter('num_episodes', 1000)
        self.declare_parameter('save_dir', './models')
        self.declare_parameter('log_dir', './logs')
        self.declare_parameter('total_timesteps', 1000000)
        self.declare_parameter('n_steps', 2048)
        
        # Create directories
        self.save_dir = self.get_parameter('save_dir').value
        self.log_dir = self.get_parameter('log_dir').value
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.save_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Logging
        self.rewards_history = []
        self.detection_confidence_history = []
        self.stability_history = []
        
        self.get_logger().info("Training Node initialized")
    
    def train(self):
        """Main training function"""
        # Create environment and policy
        self.get_logger().info("Creating environment and policy...")
        env = ModelAsModalEnv()
        
        # Create policy network
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        policy = PPOPolicy(state_dim=state_dim, action_dim=action_dim)
        
        # Create trainer
        trainer = PPOTrainer(env, policy)
        
        # Get training parameters
        total_timesteps = self.get_parameter('total_timesteps').value
        n_steps = self.get_parameter('n_steps').value
        
        self.get_logger().info(f"Starting Model as Modal training for {total_timesteps} timesteps")
        
        try:
            # Train
            episode_rewards, episode_confidences = trainer.train(
                total_timesteps=total_timesteps,
                n_steps=n_steps
            )
            
            # Save results
            self.rewards_history = episode_rewards
            self.detection_confidence_history = episode_confidences
            
            # Save final model
            final_model_path = os.path.join(self.run_dir, 'model_final.pth')
            torch.save(policy.state_dict(), final_model_path)
            self.get_logger().info(f"Final model saved: {final_model_path}")
            
            # Plot and save results
            self.plot_training_results()
            
        except KeyboardInterrupt:
            self.get_logger().info("Training interrupted by user")
            # Save current model
            interrupted_model_path = os.path.join(self.run_dir, 'model_interrupted.pth')
            torch.save(policy.state_dict(), interrupted_model_path)
            self.get_logger().info(f"Interrupted model saved: {interrupted_model_path}")
        
        finally:
            env.close()
    
    def plot_training_results(self):
        """Plot and save training results"""
        if not self.rewards_history:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot rewards
        ax1.plot(self.rewards_history)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Model as Modal Training - Episode Rewards')
        ax1.grid(True)
        
        # Calculate moving average
        window_size = min(50, len(self.rewards_history) // 10)
        if window_size > 1:
            moving_avg = np.convolve(self.rewards_history, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            ax1.plot(range(window_size-1, len(self.rewards_history)), 
                    moving_avg, 'r-', linewidth=2, label=f'{window_size}-episode average')
            ax1.legend()
        
        # Plot detection confidence
        if self.detection_confidence_history:
            ax2.plot(self.detection_confidence_history)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Mean Detection Confidence')
            ax2.set_title('ArUco Detection Confidence During Training')
            ax2.grid(True)
            
            if window_size > 1:
                conf_moving_avg = np.convolve(self.detection_confidence_history, 
                                            np.ones(window_size)/window_size, 
                                            mode='valid')
                ax2.plot(range(window_size-1, len(self.detection_confidence_history)), 
                        conf_moving_avg, 'r-', linewidth=2, label=f'{window_size}-episode average')
                ax2.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.run_dir, 'training_results.png')
        plt.savefig(plot_path)
        plt.close()
        
        self.get_logger().info(f"Training results saved: {plot_path}")
        
        # Save raw data
        import json
        data = {
            'rewards': self.rewards_history,
            'detection_confidences': self.detection_confidence_history,
            'timestamp': self.timestamp
        }
        json_path = os.path.join(self.run_dir, 'training_data.json')
        with open(json_path, 'w') as f:
            json.dump(data, f)
        self.get_logger().info(f"Training data saved: {json_path}")


def main(args=None):
    rclpy.init(args=args)
    training_node = TrainingNode()
    
    try:
        training_node.train()
    except Exception as e:
        training_node.get_logger().error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        training_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()