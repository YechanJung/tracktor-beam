#!/usr/bin/env python3
"""
model_as_modal/model_as_modal/aruco_detector.py

Enhanced ArUco detector that publishes confidence scores for RL
This is the key component that calculates detection confidence for Model as Modal
"""

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge

# ROS2 message imports
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Header
from rcl_interfaces.msg import SetParametersResult

# Constants
ARUCO_DICT = cv2.aruco.DICT_6X6_250
HOVER_HEIGHT = 2.0  # meters - lower for ArUco detection
MAX_EPISODE_STEPS = 1000


class ArUcoDetectorNode(Node):
    """
    Enhanced ArUco detector that publishes confidence scores for RL
    """
    def __init__(self):
        super().__init__('aruco_detector_modal')
        
        # Parameters
        self.declare_parameter('marker_size', 0.15)  # 15cm marker
        self.declare_parameter('camera_topic', '/camera')
        self.declare_parameter('camera_info_topic', '/camera_info')
        
        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Camera calibration
        self.camera_matrix = None
        self.dist_coeffs = None
        self.cv_bridge = CvBridge()
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter('camera_topic').value,
            self.image_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self.camera_info_callback,
            10
        )
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco_pose', 10)
        self.confidence_pub = self.create_publisher(Float32MultiArray, '/aruco_confidence', 10)
        self.detection_stats_pub = self.create_publisher(Float32MultiArray, '/model_as_modal/detection_stats', 10)
        self.annotated_image_pub = self.create_publisher(Image, '/aruco_annotated', 10)
        
        self.get_logger().info('ArUco Detector for Model as Modal initialized')
        
    def camera_info_callback(self, msg):
        """Extract camera calibration parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
        
    def image_callback(self, msg):
        """Process image and detect ArUco markers"""
        if self.camera_matrix is None:
            return
            
        # Convert to OpenCV format
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        # Calculate confidence metrics for Model as Modal
        detection_stats = self.calculate_detection_stats(corners, ids, gray)
        
        # Publish detection statistics
        stats_msg = Float32MultiArray()
        stats_msg.data = [
            detection_stats['mean_confidence'],
            detection_stats['max_confidence'],
            detection_stats['detection_count'],
            detection_stats['detection_quality']
        ]
        self.detection_stats_pub.publish(stats_msg)
        
        # If markers detected, estimate pose
        if ids is not None and len(ids) > 0:
            # Estimate pose of each marker
            marker_size = self.get_parameter('marker_size').value
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_size, self.camera_matrix, self.dist_coeffs
            )
            
            # Publish pose of first marker (for simplicity)
            pose_msg = PoseStamped()
            pose_msg.header.stamp = msg.header.stamp
            pose_msg.header.frame_id = 'camera_frame'
            
            # Convert rotation vector to quaternion
            rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
            quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
            
            pose_msg.pose.position.x = tvecs[0][0][0]
            pose_msg.pose.position.y = tvecs[0][0][1]
            pose_msg.pose.position.z = tvecs[0][0][2]
            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            pose_msg.pose.orientation.w = quaternion[3]
            
            self.pose_pub.publish(pose_msg)
            
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            
        # Publish annotated image
        annotated_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.annotated_image_pub.publish(annotated_msg)
    
    def calculate_detection_stats(self, corners, ids, gray_image):
        """
        Calculate detection statistics for Model as Modal reward
        This is the key innovation - using detection confidence in RL
        """
        stats = {
            'mean_confidence': 0.0,
            'max_confidence': 0.0,
            'detection_count': 0,
            'detection_quality': 0.0
        }
        
        if ids is None or len(ids) == 0:
            return stats
            
        confidences = []
        
        for i, corner in enumerate(corners):
            # Calculate confidence based on multiple factors
            
            # 1. Corner sharpness (how well-defined the corners are)
            corner_sharpness = self.calculate_corner_sharpness(corner[0], gray_image)
            
            # 2. Marker size consistency (how square the marker appears)
            size_consistency = self.calculate_size_consistency(corner[0])
            
            # 3. Contrast ratio (marker vs background)
            contrast_ratio = self.calculate_contrast_ratio(corner[0], gray_image)
            
            # Combined confidence score
            confidence = (corner_sharpness * 0.4 + 
                         size_consistency * 0.3 + 
                         contrast_ratio * 0.3)
            
            confidences.append(confidence)
        
        stats['mean_confidence'] = float(np.mean(confidences))
        stats['max_confidence'] = float(np.max(confidences))
        stats['detection_count'] = len(ids)
        stats['detection_quality'] = stats['mean_confidence'] * min(len(ids), 5) / 5.0
        
        return stats
    
    def calculate_corner_sharpness(self, corners, gray_image):
        """Calculate how sharp/well-defined the corners are"""
        sharpness_scores = []
        
        for corner in corners:
            x, y = int(corner[0]), int(corner[1])
            # Use Harris corner detector
            window = gray_image[max(0, y-5):min(gray_image.shape[0], y+5),
                               max(0, x-5):min(gray_image.shape[1], x+5)]
            if window.size > 0:
                harris = cv2.cornerHarris(window.astype(np.float32), 2, 3, 0.04)
                sharpness_scores.append(np.max(harris))
        
        return np.mean(sharpness_scores) / 10.0 if sharpness_scores else 0.0
    
    def calculate_size_consistency(self, corners):
        """Calculate how square/consistent the marker appears"""
        # Calculate side lengths
        sides = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            sides.append(np.linalg.norm(p2 - p1))
        
        # Calculate variance in side lengths
        variance = np.var(sides)
        mean_side = np.mean(sides)
        
        # Lower variance = more consistent = higher score
        consistency = 1.0 / (1.0 + variance / mean_side**2)
        return consistency
    
    def calculate_contrast_ratio(self, corners, gray_image):
        """Calculate contrast between marker and background"""
        # Create mask for marker region
        mask = np.zeros(gray_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [corners.astype(int)], 255)
        
        # Calculate mean intensity inside and outside marker
        inside_mean = cv2.mean(gray_image, mask=mask)[0]
        outside_mask = cv2.bitwise_not(mask)
        outside_mean = cv2.mean(gray_image, mask=outside_mask)[0]
        
        # Contrast ratio
        contrast = abs(inside_mean - outside_mean) / 255.0
        return contrast
    
    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
                
        return np.array([x, y, z, w])


def main(args=None):
    rclpy.init(args=args)
    detector = ArUcoDetectorNode()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()