# Model as Modal: ArUco-based Drone Hovering with RL

This package implements the "Model as Modal" approach for drone hovering using ArUco marker detection confidence integrated into the reinforcement learning reward function.

## Overview

The Model as Modal approach treats the perception model (ArUco detection) as a modal component in the RL state and reward function. This allows the drone to learn behaviors that optimize both physical stability and visual marker detection quality.

## Directory Structure

```
model_as_modal/
├── model_as_modal/         # Python package
│   ├── __init__.py
│   ├── environment.py      # Gymnasium environment
│   ├── aruco_detector.py   # Enhanced ArUco detector
│   ├── policy.py          # PPO policy network
│   └── trainer.py         # PPO training algorithm
├── scripts/               # Executable scripts
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── launch/               # ROS2 launch files
│   └── model_as_modal.launch.py
├── config/              # Configuration files
│   └── model_as_modal.yaml
├── resource/            # ROS2 resource files
├── package.xml          # ROS2 package manifest
├── setup.py            # Python setup file
├── setup.cfg           # Setup configuration
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Installation

1. Clone the repository into your ROS2 workspace:

```bash
cd ~/ros2_ws/src
git clone https://github.com/ARK-Electronics/tracktor-beam.git
git clone  model_as_modal
```

2. Install Python dependencies:

```bash
cd model_as_modal
pip install -r requirements.txt
```

3. Build the workspace:

```bash
cd ~/ros2_ws
colcon build --packages-select model_as_modal
source install/setup.bash
```

## Usage

### Training

Launch the complete training system:

```bash
ros2 launch model_as_modal model_as_modal.launch.py
```

Or run components separately:

1. Start the ArUco detector:

```bash
ros2 run model_as_modal aruco_detector
```

2. Start the training script:

```bash
ros2 run model_as_modal train
```

### Evaluation

Evaluate a trained model:

```bash
ros2 run model_as_modal evaluate --ros-args -p model_path:=./models/model_final.pth
```

### Visualization

Visualize training results:

```bash
ros2 run model_as_modal visualize
```

## Key Features

1. **Model as Modal Integration**: ArUco detection confidence is integrated directly into the RL state and reward function
2. **Gymnasium Compatibility**: Standard gym.Env interface for easy integration with RL libraries
3. **PPO Implementation**: Custom PPO algorithm optimized for the drone hovering task
4. **Enhanced ArUco Detection**: Multiple confidence metrics (corner sharpness, size consistency, contrast ratio)
5. **ROS2 Integration**: Full integration with PX4 through ROS2

## Configuration

Edit `config/model_as_modal.yaml` to adjust:

- ArUco detection parameters
- Environment settings
- Reward weights (key for Model as Modal)
- PPO hyperparameters
- Training parameters

## Research Innovation

The Model as Modal approach creates a multimodal state representation that combines:

- **Physical State**: Position, velocity, angular velocity (9 dimensions)
- **Perception State**: Detection confidence metrics (4 dimensions)

This allows the drone to learn hovering behaviors that balance:

- Physical stability
- Visual marker detection quality
- Energy efficiency

## Citation

If you use this code in your research, please cite:

```
@article{your2025model,
  title={Model as Modal: Integrating Perception Confidence in Drone Reinforcement Learning},
  author={Your Name},
  journal={Your Conference/Journal},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details
