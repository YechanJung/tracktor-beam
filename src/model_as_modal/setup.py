from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'model_as_modal'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
            glob(os.path.join('launch', '*.py'))),
        (os.path.join('share', package_name, 'config'), 
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Model as Modal: RL-based drone hovering with ArUco detection confidence',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_detector = model_as_modal.aruco_detector:main',
            'environment = model_as_modal.environment:main',
            'trainer = model_as_modal.trainer:main',
            'train = scripts.train:main',
            'evaluate = scripts.evaluate:main',
            'visualize = scripts.visualize:main',
        ],
    },
)