"""
model_as_modal/model_as_modal/environment.py

Main Gymnasium environment for Model as Modal
Combines gym.Env interface with ROS2 Node for drone control
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import threading
import time

# ROS2 message imports
from px4_msgs.msg import (
    VehicleLocalPosition, 
    VehicleAttitude,
    VehicleCommand,
    OffboardControlMode,
    TrajectorySetpoint
)
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped

# Constants
HOVER_HEIGHT = 2.0  # meters
MAX_EPISODE_STEPS = 1000


class ModelAsModalEnv(gym.Env, Node):
    """
    Gymnasium Environment for Model as Modal
    Combines gym.Env interface with ROS2 Node for drone control
    """
    
    metadata = {'render_modes': ['console']}
    
    def __init__(self, node_name='model_as_modal_env'):
        # Initialize ROS2 Node
        Node.__init__(self, node_name)
        
        # Initialize Gym Environment
        super(ModelAsModalEnv, self).__init__()
        
        # Define action and observation spaces
        # Action space: [vx, vy, vz, yaw_rate] normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-np.ones(4),
            high=np.ones(4),
            dtype=np.float32
        )
        
        # Observation space: [position(3), velocity(3), angular_velocity(3), detection_stats(4)]
        # Total: 13 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(13),
            high=np.inf * np.ones(13),
            dtype=np.float32
        )
        
        # QoS profile for PX4
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        # State variables
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude = np.zeros(4)  # quaternion
        self.angular_velocity = np.zeros(3)
        self.detection_stats = np.zeros(4)  # Model as Modal key feature
        
        # Episode variables
        self.current_step = 0
        self.target_position = np.array([0.0, 0.0, HOVER_HEIGHT])
        self.episode_start_time = None
        
        # Reward weights (Model as Modal parameters)
        self.w_stability = 0.4
        self.w_detection = 0.4
        self.w_energy = 0.2
        
        # Setup ROS2 subscribers and publishers
        self._setup_ros2_interfaces()
        
        # Start ROS2 spinning in separate thread
        self.ros_thread = threading.Thread(target=self._ros_spin, daemon=True)
        self.ros_thread.start()
        
        # Wait for initial state
        time.sleep(0.5)
        
        self.get_logger().info('Model as Modal Gym Environment initialized')
    
    def _setup_ros2_interfaces(self):
        """Setup ROS2 subscribers and publishers"""
        # Subscribers - PX4 topics
        self.position_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self._position_callback,
            self.qos_profile
        )
        
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self._attitude_callback,
            self.qos_profile
        )
        
        # Subscriber - ArUco detection stats (Model as Modal)
        self.detection_sub = self.create_subscription(
            Float32MultiArray,
            '/model_as_modal/detection_stats',
            self._detection_callback,
            10
        )
        
        # Publishers - PX4 control
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            self.qos_profile
        )
        
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            self.qos_profile
        )
        
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            self.qos_profile
        )
    
    def _ros_spin(self):
        """Spin ROS2 in separate thread"""
        rclpy.spin(self)
    
    def _position_callback(self, msg):
        """Update position and velocity from PX4"""
        self.position[0] = msg.x
        self.position[1] = msg.y
        self.position[2] = -msg.z  # PX4 uses NED, convert to NEU
        
        self.velocity[0] = msg.vx
        self.velocity[1] = msg.vy
        self.velocity[2] = -msg.vz
    
    def _attitude_callback(self, msg):
        """Update attitude from PX4"""
        self.attitude[0] = msg.q[0]
        self.attitude[1] = msg.q[1]
        self.attitude[2] = msg.q[2]
        self.attitude[3] = msg.q[3]
        
        # Angular velocity
        self.angular_velocity[0] = msg.rollspeed
        self.angular_velocity[1] = -msg.pitchspeed
        self.angular_velocity[2] = -msg.yawspeed
    
    def _detection_callback(self, msg):
        """Update ArUco detection statistics - Key for Model as Modal"""
        self.detection_stats = np.array(msg.data)
    
    def _get_observation(self):
        """Get current observation combining physical and perception state"""
        # Physical state (9 dimensions)
        physical_state = np.concatenate([
            self.position,
            self.velocity,
            self.angular_velocity
        ])
        
        # Detection state (4 dimensions) - Model as Modal innovation
        perception_state = self.detection_stats
        
        # Combined state (13 dimensions)
        observation = np.concatenate([physical_state, perception_state])
        
        return observation.astype(np.float32)
    
    def reset(self, *, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset episode variables
        self.current_step = 0
        self.episode_start_time = time.time()
        
        # Command drone to initial position
        self._command_position(self.target_position)
        
        # Wait for drone to reach position
        time.sleep(2.0)
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute action and return new state, reward, terminated, truncated, info"""
        # Validate action
        action = np.clip(action, -1, 1)
        
        # Scale actions to appropriate ranges
        velocity_command = action[:3] * 2.0  # Max 2 m/s
        yaw_rate = action[3] * 1.0  # Max 1 rad/s
        
        # Send command to PX4
        self._send_velocity_command(velocity_command, yaw_rate)
        
        # Wait for state update (simulate control frequency)
        time.sleep(0.05)  # 20 Hz control
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward (Model as Modal reward function)
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._check_terminated()
        
        # Check truncation (max steps)
        self.current_step += 1
        truncated = self.current_step >= MAX_EPISODE_STEPS
        
        # Get info
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self):
        """
        Calculate Model as Modal reward
        Combines hovering stability with ArUco detection confidence
        """
        # 1. Stability component
        position_error = np.linalg.norm(self.position - self.target_position)
        angular_rate = np.linalg.norm(self.angular_velocity)
        velocity_magnitude = np.linalg.norm(self.velocity)
        
        stability_reward = (
            -position_error * 1.0 -
            angular_rate * 0.5 -
            velocity_magnitude * 0.3
        )
        
        # 2. Detection component - Model as Modal key innovation
        mean_confidence = self.detection_stats[0]
        max_confidence = self.detection_stats[1]
        detection_count = self.detection_stats[2]
        detection_quality = self.detection_stats[3]
        
        if detection_count > 0:
            detection_reward = (
                mean_confidence * 3.0 +
                max_confidence * 1.5 +
                detection_quality * 2.0
            )
        else:
            detection_reward = -3.0  # Penalty for no detection
        
        # 3. Energy component
        control_effort = np.linalg.norm(self.velocity) + abs(self.angular_velocity[2])
        energy_reward = -control_effort * 0.1
        
        # Combined reward with Model as Modal weights
        total_reward = (
            self.w_stability * stability_reward +
            self.w_detection * detection_reward +
            self.w_energy * energy_reward
        )
        
        return float(total_reward)
    
    def _check_terminated(self):
        """Check if episode should terminate"""
        # Terminate if drone is too far from target
        position_error = np.linalg.norm(self.position - self.target_position)
        if position_error > 5.0:
            return True
        
        # Terminate if drone crashes (z < 0.1)
        if self.position[2] < 0.1:
            return True
        
        # Terminate if lost visual contact for too long
        if self.detection_stats[2] == 0:  # No detections
            # Count consecutive steps without detection
            if not hasattr(self, 'no_detection_steps'):
                self.no_detection_steps = 0
            self.no_detection_steps += 1
            
            if self.no_detection_steps > 100:  # 5 seconds at 20Hz
                return True
        else:
            self.no_detection_steps = 0
        
        return False
    
    def _get_info(self):
        """Get additional information about current state"""
        info = {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'angular_velocity': self.angular_velocity.copy(),
            'detection_stats': self.detection_stats.copy(),
            'detection_confidence': self.detection_stats[0],
            'detection_count': int(self.detection_stats[2]),
            'position_error': np.linalg.norm(self.position - self.target_position),
            'episode_time': time.time() - self.episode_start_time if self.episode_start_time else 0
        }
        
        # Calculate reward components for analysis
        info['reward_components'] = {
            'stability': self._calculate_stability_reward(),
            'detection': self._calculate_detection_reward(),
            'energy': self._calculate_energy_reward()
        }
        
        return info
    
    def _calculate_stability_reward(self):
        """Calculate stability component of reward"""
        position_error = np.linalg.norm(self.position - self.target_position)
        angular_rate = np.linalg.norm(self.angular_velocity)
        velocity_magnitude = np.linalg.norm(self.velocity)
        
        return -(position_error * 1.0 + angular_rate * 0.5 + velocity_magnitude * 0.3)
    
    def _calculate_detection_reward(self):
        """Calculate detection component of reward"""
        if self.detection_stats[2] > 0:
            return (self.detection_stats[0] * 3.0 + 
                   self.detection_stats[1] * 1.5 + 
                   self.detection_stats[3] * 2.0)
        return -3.0
    
    def _calculate_energy_reward(self):
        """Calculate energy component of reward"""
        control_effort = np.linalg.norm(self.velocity) + abs(self.angular_velocity[2])
        return -control_effort * 0.1
    
    def _send_velocity_command(self, velocity, yaw_rate):
        """Send velocity command to PX4"""
        # Enable offboard mode
        offboard_msg = OffboardControlMode()
        offboard_msg.position = False
        offboard_msg.velocity = True
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_pub.publish(offboard_msg)
        
        # Send velocity setpoint
        trajectory_msg = TrajectorySetpoint()
        trajectory_msg.position = [float('nan'), float('nan'), float('nan')]
        trajectory_msg.velocity = [velocity[0], velocity[1], -velocity[2]]  # Convert to NED
        trajectory_msg.yaw = float('nan')
        trajectory_msg.yawspeed = -yaw_rate  # Convert to NED
        trajectory_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(trajectory_msg)
    
    def _command_position(self, position):
        """Send position command for reset"""
        trajectory_msg = TrajectorySetpoint()
        trajectory_msg.position = [position[0], position[1], -position[2]]  # Convert to NED
        trajectory_msg.velocity = [0.0, 0.0, 0.0]
        trajectory_msg.yaw = 0.0
        trajectory_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(trajectory_msg)
    
    def render(self):
        """Render environment (console output for now)"""
        print(f"Step: {self.current_step}")
        print(f"Position: {self.position}")
        print(f"Detection Confidence: {self.detection_stats[0]:.3f}")
        print(f"Position Error: {np.linalg.norm(self.position - self.target_position):.3f}")
        print("---")
    
    def close(self):
        """Clean up resources"""
        self.get_logger().info("Closing Model as Modal Environment")
        # The ROS2 node will be cleaned up by the executor