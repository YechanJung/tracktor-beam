import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Get package directories
    pkg_model_as_modal = get_package_share_directory('model_as_modal')
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    config_file = LaunchConfiguration('config_file', default=os.path.join(
        pkg_model_as_modal, 'config', 'model_as_modal.yaml'))
    
    # Model as Modal ArUco detector (enhanced version)
    modal_aruco_detector = Node(
        package='model_as_modal',
        executable='aruco_detector',
        name='aruco_detector_modal',
        parameters=[config_file, {'use_sim_time': use_sim_time}],
        remappings=[
            ('/camera', '/camera'),
            ('/camera_info', '/camera_info'),
        ],
        output='screen'
    )
    
    # Model as Modal training script
    modal_trainer = Node(
        package='model_as_modal',
        executable='train',
        name='model_as_modal_trainer',
        parameters=[{
            'num_episodes': 1000,
            'save_dir': './models',
            'log_dir': './logs',
            'total_timesteps': 1000000,
            'n_steps': 2048,
        }],
        output='screen'
    )
    
    # ROS2 bag recording for analysis
    bag_record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '-o', 'model_as_modal_experiment',
            '/aruco_pose',
            '/aruco_confidence',
            '/model_as_modal/detection_stats',
            '/fmu/out/vehicle_local_position',
            '/fmu/out/vehicle_attitude',
            '/aruco_annotated',
        ],
        output='screen'
    )
    
    # RViz for visualization (optional)
    rviz_config_file = os.path.join(pkg_model_as_modal, 'config', 'model_as_modal.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        condition=LaunchConfiguration('use_rviz', default='false')
    )
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('config_file', default_value=config_file),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        
        # Launch nodes
        modal_aruco_detector,
        modal_trainer,
        bag_record,
        rviz,
    ])